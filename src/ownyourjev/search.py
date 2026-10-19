"""Training-strategy search: which backbone, which method, how many labels, which hyperparameters.

Stage A  frozen-embedding heads for every backbone on the full data-size ladder (seconds each).
Stage B  LoRA and full fine-tuning for every backbone on a reduced ladder with backbone-default recipes.
Stage C  successive-halving hyperparameter search (Optuna, median pruning) on the best (backbone, method)
         at the largest size, under a GPU-minute cap.

Selection always uses the held-out *silver* validation set. Any ``test_fn`` metrics are recorded for
reporting only. Every trial is appended to ``<out_dir>/search.jsonl`` so a run can be resumed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ownyourjev.data import Example, stratified_subsample
from ownyourjev.evaluate import primary_metric_name
from ownyourjev.spec import Question
from ownyourjev.train import TrainConfig, TrainResult, train_with_oom_retry


@dataclass
class SearchConfig:
    backbones: list[str] = field(default_factory=lambda: ["ultrabert", "ultrabert-embed", "gte-modernbert"])
    methods: list[str] = field(default_factory=lambda: ["frozen", "lora", "full"])
    ladder_frozen: list[int] = field(default_factory=lambda: [50, 100, 250, 500, 1000, 2000, 5000])
    ladder_ft: list[int] = field(default_factory=lambda: [100, 500, 2000, 5000])
    hpo_trials: int = 10
    hpo_gpu_minutes: float = 20.0
    max_len: int = 512
    batch_size: int = 16
    grad_accum: int = 1
    seed: int = 0
    device: str = "auto"
    final_seeds: int = 3

    def epochs_for(self, n: int, method: str) -> int:
        if method == "frozen":
            return 0
        return 10 if n <= 500 else (5 if n <= 2000 else 3)


@dataclass
class Trial:
    stage: str
    backbone: str
    method: str
    n_train: int
    config: dict
    val_loss: float
    val_metric: float
    val_metrics: dict
    test_metrics: dict | None
    train_s: float
    wall_s: float
    best_epoch: int
    temperature: float
    seed: int = 0
    key: str = ""

    @staticmethod
    def make_key(stage: str, backbone: str, method: str, n: int, seed: int, extra: str = "") -> str:
        return f"{stage}|{backbone}|{method}|{n}|{seed}|{extra}"


def _rank(t: Trial) -> tuple[float, float, float]:
    """Selection order on silver validation only: metric, then lower loss, then less training time."""
    m = t.val_metric if math.isfinite(t.val_metric) else -1.0
    loss = -t.val_loss if math.isfinite(t.val_loss) else -1e9
    return (round(m, 4), loss, -t.train_s)


class Search:
    def __init__(
        self,
        question: Question,
        qid: str,
        train_pool: list[Example],
        val: list[Example],
        cfg: SearchConfig,
        out_dir: str | Path,
        *,
        test_fn: Callable[[TrainResult], dict] | None = None,
        log: Callable[[str], None] = print,
        metric: str | None = None,
    ):
        self.question, self.qid, self.pool, self.val, self.cfg = question, qid, train_pool, val, cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.test_fn = test_fn
        self.log = log
        self.metric = metric or primary_metric_name(question)
        self.trials: list[Trial] = []
        self.best: tuple[float, Trial, TrainResult | None] | None = None
        self._load()

    # ---- persistence ----------------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.out_dir / "search.jsonl"

    def _load(self) -> None:
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.trials.append(Trial(**json.loads(line)))

    def _append(self, t: Trial) -> None:
        self.trials.append(t)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(t)) + "\n")

    def _have(self, key: str) -> Trial | None:
        return next((t for t in self.trials if t.key == key), None)

    # ---- one trial --------------------------------------------------------------------------------
    def run_trial(self, stage: str, tcfg: TrainConfig, n: int, *, extra: str = "", keep_model: bool = True) -> Trial:
        key = Trial.make_key(stage, tcfg.backbone, tcfg.method, n, tcfg.seed, extra)
        have = self._have(key)
        if have is not None:
            return have
        sub = stratified_subsample(self.pool, n, seed=tcfg.seed) if n < len(self.pool) else self.pool
        res = self._train_with_oom_retry(sub, tcfg)
        if res is None:  # gave up: record a failed trial so the search can continue and never retries it
            t = Trial(stage, tcfg.backbone, tcfg.method, len(sub), tcfg.to_dict(), math.nan, math.nan, {}, None, 0.0, 0.0, 0, 1.0, tcfg.seed, key)
            self._append(t)
            self.log(f"[{stage}] {tcfg.backbone:16s} {tcfg.method:6s} n={len(sub):5d} FAILED (out of memory twice)")
            return t
        test = self.test_fn(res) if self.test_fn else None
        val_metric = res.val_metrics.get(self.metric, res.val_metric)
        t = Trial(stage, tcfg.backbone, tcfg.method, len(sub), tcfg.to_dict(), res.val_loss, val_metric, res.val_metrics, test, res.train_s, res.wall_s, res.best_epoch, res.temperature, tcfg.seed, key)
        self._append(t)
        self.log(f"[{stage}] {tcfg.backbone:16s} {tcfg.method:6s} n={len(sub):5d} val_{self.metric}={val_metric:.4f} val_loss={res.val_loss:.4f}" + (f" test={test.get(self.metric, float('nan')):.4f}" if test else "") + f" train_s={res.train_s:.0f}")
        if keep_model and (self.best is None or _rank(t) > _rank(self.best[1])):
            self.best = (t.val_metric, t, res)
            res.model.save(self.out_dir / "best_model", res.tokenizer)
            (self.out_dir / "best_model" / "trial_key.txt").write_text(key)
            res.model.to("cpu")  # keep the GPU free for the next trials; moved back when it is used
        del res
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        return t

    def _train_with_oom_retry(self, sub: list[Example], tcfg: TrainConfig) -> TrainResult | None:
        return train_with_oom_retry(self.question, sub, self.val, tcfg, qid=self.qid, log=self.log)

    def _tcfg(self, backbone: str, method: str, n: int, **kw) -> TrainConfig:
        base = dict(backbone=backbone, method=method, epochs=self.cfg.epochs_for(n, method), batch_size=self.cfg.batch_size, grad_accum=self.cfg.grad_accum, max_len=self.cfg.max_len, seed=self.cfg.seed, device=self.cfg.device)
        base.update(kw)
        return TrainConfig(**base)

    # ---- stages -----------------------------------------------------------------------------------
    def stage_a(self) -> None:
        if "frozen" not in self.cfg.methods:
            return
        for bb in self.cfg.backbones:
            for n in self.cfg.ladder_frozen:
                if n > len(self.pool) and n != self.cfg.ladder_frozen[-1]:
                    continue
                self.run_trial("A", self._tcfg(bb, "frozen", min(n, len(self.pool))), min(n, len(self.pool)))

    def stage_b(self) -> None:
        for method in [m for m in self.cfg.methods if m != "frozen"]:
            for bb in self.cfg.backbones:
                for n in self.cfg.ladder_ft:
                    if n > len(self.pool) and n != self.cfg.ladder_ft[-1]:
                        continue
                    self.run_trial("B", self._tcfg(bb, method, min(n, len(self.pool))), min(n, len(self.pool)))

    def best_setting(self) -> tuple[str, str]:
        """Best (backbone, method) by silver-val metric at the largest size seen."""
        n_max = max(t.n_train for t in self.trials)
        cands = [t for t in self.trials if t.n_train == n_max and t.stage in ("A", "B")]
        best = max(cands, key=_rank)
        return best.backbone, best.method

    def stage_c(self) -> None:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        bb, method = self.best_setting()
        n = min(self.cfg.ladder_ft[-1], len(self.pool))
        if method == "frozen":
            self.log("[C] best method is frozen: no gradient HPO needed")
            return
        from ownyourjev.models.backbones import resolve_backbone

        spec = resolve_backbone(bb)
        base_lr = spec.lora_lr if method == "lora" else spec.default_lr
        budget_s = self.cfg.hpo_gpu_minutes * 60
        spent = 0.0
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.cfg.seed), pruner=optuna.pruners.MedianPruner(n_startup_trials=3))

        def objective(trial: optuna.Trial) -> float:
            nonlocal spent
            if spent > budget_s:
                raise optuna.TrialPruned()
            kw = dict(
                lr=trial.suggest_float("lr", base_lr / 4, base_lr * 4, log=True),
                epochs=trial.suggest_categorical("epochs", [3, 5, 8]),
                warmup_frac=trial.suggest_categorical("warmup_frac", [0.0, 0.06, 0.1]),
                label_smoothing=trial.suggest_categorical("label_smoothing", [0.0, 0.1]) if self.question.type == "choice" else 0.0,
                ordinal_sigma=trial.suggest_categorical("ordinal_sigma", [0.0, 0.5, 1.0]) if self.question.type == "score" else 0.0,
                batch_size=trial.suggest_categorical("batch_size", [16, 32]),
                weight_decay=trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1]),
            )
            if method == "lora":
                kw["lora_r"] = trial.suggest_categorical("lora_r", [8, 16, 32])
            tcfg = self._tcfg(bb, method, n, **kw)
            t = self.run_trial("C", tcfg, n, extra=f"t{trial.number}")
            spent += t.train_s
            return t.val_metric if math.isfinite(t.val_metric) else -1.0

        study.optimize(objective, n_trials=self.cfg.hpo_trials, catch=(RuntimeError,))
        self.log(f"[C] best params {study.best_params} val_{self.metric}={study.best_value:.4f} (gpu-min used {spent / 60:.1f})")

    def stage_final(self) -> list[Trial]:
        """Re-train the selected recipe with several seeds at the largest size for variance."""
        sel = self.selected()
        out = []
        for seed in range(1, self.cfg.final_seeds):
            tcfg = TrainConfig(**{**sel.config, "seed": seed})
            out.append(self.run_trial("F", tcfg, sel.n_train))  # a better-validating seed may become the selection
        return [sel] + out

    def selected(self) -> Trial:
        return self.selected_from("A", "B", "C", "F")

    def best_model_on_disk(self, key: str | None = None) -> Path | None:
        """The saved model of the trial ``key`` (default: the selected trial), if that is what is on disk."""
        d = self.out_dir / "best_model"
        k = d / "trial_key.txt"
        key = key or self.selected().key
        if k.is_file() and k.read_text().strip() == key and (d / "meta.json").is_file():
            return d
        return None

    def selected_from(self, *stages: str) -> Trial:
        cands = [t for t in self.trials if t.stage in stages]
        return max(cands, key=_rank)

    def run(self, stages: str = "ABCF") -> dict:
        if "A" in stages:
            self.stage_a()
        if "B" in stages:
            self.stage_b()
        if "C" in stages:
            self.stage_c()
        finals = self.stage_final() if "F" in stages else [self.selected()]
        sel = self.selected()
        report = {
            "selected": asdict(sel),
            "final_seeds": [asdict(t) for t in finals],
            "n_trials": len(self.trials),
            "gpu_seconds_total": sum(t.train_s for t in self.trials),
            "learning_curves": self.learning_curves(),
            "best_by_setting": self.best_by_setting(),
        }
        (self.out_dir / "report.json").write_text(json.dumps(report, indent=1))
        return report

    # ---- summaries --------------------------------------------------------------------------------
    def learning_curves(self, stages: tuple[str, ...] = ("A", "B")) -> dict[str, list[dict]]:
        curves: dict[str, list[dict]] = {}
        for t in self.trials:
            if t.stage not in stages:
                continue
            curves.setdefault(f"{t.backbone}/{t.method}", []).append({"n": t.n_train, "val": t.val_metric, "test": (t.test_metrics or {}).get(self.metric), "train_s": t.train_s})
        for v in curves.values():
            v.sort(key=lambda r: r["n"])
        return curves

    def best_by_setting(self) -> list[dict]:
        rows = {}
        for t in self.trials:
            k = (t.backbone, t.method)
            if k not in rows or t.val_metric > rows[k].val_metric:
                rows[k] = t
        out = [{"backbone": b, "method": m, "n": t.n_train, "val": t.val_metric, "test": (t.test_metrics or {}).get(self.metric), "train_s": t.train_s, "stage": t.stage} for (b, m), t in rows.items()]
        return sorted(out, key=lambda r: -(r["val"] if r["val"] == r["val"] else -1))
