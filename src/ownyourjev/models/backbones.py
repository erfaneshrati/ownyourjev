"""Registry of supported encoder backbones and their sane defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    pooling: str  # "cls" | "mean"
    default_lr: float  # for full fine-tuning
    lora_lr: float
    max_len: int
    lora_targets: list[str] = field(default_factory=list)
    notes: str = ""


BACKBONES: dict[str, BackboneSpec] = {
    "ultrabert": BackboneSpec(
        name="ultrabert",
        hf_id="dmis-lab/UltraBERT-150M",
        pooling="cls",
        default_lr=1e-5,  # UltraBERT authors: ~10x lower LR than other encoders
        lora_lr=2e-4,
        max_len=32768,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        notes="Base pre-trained encoder; recommended by its authors for classification fine-tuning.",
    ),
    "ultrabert-embed": BackboneSpec(
        name="ultrabert-embed",
        hf_id="dmis-lab/UltraBERT-Embed-150M",
        pooling="cls",
        default_lr=1e-5,
        lora_lr=2e-4,
        max_len=32768,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        notes="Contrastively trained embedding variant; strong for frozen-embedding heads.",
    ),
    "gte-modernbert": BackboneSpec(
        name="gte-modernbert",
        hf_id="Alibaba-NLP/gte-modernbert-base",
        pooling="cls",
        default_lr=3e-5,
        lora_lr=3e-4,
        max_len=8192,
        lora_targets=["Wqkv", "Wo", "Wi"],
        notes="GTE embedding model on ModernBERT-base.",
    ),
    "modernbert-large": BackboneSpec(
        name="modernbert-large",
        hf_id="answerdotai/ModernBERT-large",
        pooling="cls",
        default_lr=2e-5,
        lora_lr=2e-4,
        max_len=8192,
        lora_targets=["Wqkv", "Wo", "Wi"],
        notes="395M. CPU int8 p95 (8 threads, loaded machine): ~44 ms @32 tok, ~64 ms @64 tok, ~100 ms @128 tok. "
        "Only viable for short-input tasks under the 70 ms gate.",
    ),
}


def resolve_backbone(name: str) -> BackboneSpec:
    if name in BACKBONES:
        return BACKBONES[name]
    for spec in BACKBONES.values():
        if spec.hf_id == name:
            return spec
    # unknown HF id: generic defaults
    return BackboneSpec(name=name, hf_id=name, pooling="cls", default_lr=2e-5, lora_lr=2e-4, max_len=512)
