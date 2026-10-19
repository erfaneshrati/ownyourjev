"""The 15 benchmark tasks: dataset loading, state construction and the frozen Jev question templates.

Templates, gold mappings and splits are reproduced verbatim from
https://github.com/AppliedMachineLearning-Lab/jev-benchmarking (MIT License, Copyright (c) 2026 Tobias Deußer,
Lorenz Sparrenberg, Rafet Sifa), so that (a) our numbers are directly comparable to the published Jev-1.13.0
results and (b) the paper's raw Jev responses can be looked up per example via ``request_key``.

Each task yields ``BenchExample``s with the raw question dicts (for the Jev key) and a parsed ``Question``
per head. ``pool`` = unlabeled training texts (gold kept only for the upper-bound / seeding rows),
``eval`` = the published evaluation split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from datasets import load_dataset

from ownyourjev.data import Example
from ownyourjev.spec import Question

JEV_MODEL = "jev-1.13.0"


# ---- raw question constructors (same key order as the paper's, for request hashing) ----------------
def choice(instructions: Any, criteria: dict[str, Any]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: Any, true: str | None = None, false: str | None = None) -> dict:
    q: dict = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v}
    return q


def score(instructions: Any, levels: list[Any]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def request_key(state: Any, questions: dict, model: str = JEV_MODEL) -> str:
    payload = json.dumps({"model": model, "state": state, "questions": questions}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class TaskSpec:
    name: str
    mode: str  # noul | choice | score (the primary head's type)
    primary_head: str
    primary_metric: str
    questions: dict[str, dict]  # raw question dicts, fixed per task (may be overridden per example)
    grouped: bool = False  # grouped metrics / grouped cross-fit (SummEval)
    cross_fit_folds: int = 0  # >0 when there is no separate training pool
    long_input: bool = False
    description: str = ""

    @property
    def parsed(self) -> dict[str, Question]:
        return {qid: Question(**q) for qid, q in self.questions.items()}


class Task:
    spec: TaskSpec

    def pool(self) -> list[Example]:  # unlabeled (gold retained in .gold)
        raise NotImplementedError

    def eval(self) -> list[Example]:
        raise NotImplementedError

    @staticmethod
    def _ex(uid: str, state: Any, gold: dict, group: str = "", meta: dict | None = None) -> Example:
        return Example(uid=uid, state=state, labels=None, gold=gold, group=group, meta=meta or {})


# ============================================ NOUL ==================================================
class SMSSpam(Task):
    spec = TaskSpec(
        "sms_spam", "noul", "answer", "f1",
        {"answer": noul("Is the SMS `message` spam, such as unsolicited advertising, prize scams or promotional bulk messaging?")},
        cross_fit_folds=2, description="SMS spam detection (only one split: 2-fold cross-fit).",
    )

    def _all(self) -> list[Example]:
        ds = load_dataset("ucirvine/sms_spam", "plain_text", split="train")
        return [self._ex(f"sms-{i}", {"message": r["sms"].strip()}, {"answer": r["label"] == 1}) for i, r in enumerate(ds)]

    def pool(self):
        return self._all()

    def eval(self):
        return self._all()


class PAWS(Task):
    spec = TaskSpec(
        "paws", "noul", "answer", "accuracy",
        {"answer": noul(
            "Do `sentence_1` and `sentence_2` mean the same thing?",
            true="They are paraphrases: the same meaning, possibly with different wording or word order.",
            false="Their meanings differ, even if they share most of their words.",
        )},
        description="Adversarial paraphrase identification.",
    )

    def _rows(self, split):
        ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
        return [self._ex(f"paws-{split}-{i}", {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]}, {"answer": r["label"] == 1}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


class BoolQ(Task):
    spec = TaskSpec(
        "boolq", "noul", "answer", "accuracy",
        {"answer": noul("According to `passage`, is the answer to `question` yes?")},
        long_input=True, description="Yes/no questions over Wikipedia passages.",
    )

    def _rows(self, split):
        ds = load_dataset("google/boolq", split=split)
        out = []
        for i, r in enumerate(ds):
            q = r["question"].strip()
            q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
            out.append(self._ex(f"boolq-{split}-{i}", {"passage": r["passage"], "question": q}, {"answer": bool(r["answer"])}))
        return out

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("validation")


class PromptInjections(Task):
    spec = TaskSpec(
        "prompt_injections", "noul", "answer", "accuracy",
        {"answer": noul("Is `text` a prompt injection: an attempt to override, ignore or hijack the instructions of an AI system?")},
        description="Prompt-injection detection (English/German); tiny train split -> synthetic data showcase.",
    )

    def _rows(self, split):
        ds = load_dataset("deepset/prompt-injections", split=split)
        return [self._ex(f"pi-{split}-{i}", {"text": r["text"]}, {"answer": r["label"] == 1}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


class ToxicChat(Task):
    spec = TaskSpec(
        "toxic_chat", "noul", "toxic", "f1",
        {
            "toxic": noul("Is the user's `prompt` to an AI chatbot toxic: harmful, offensive or sexual, or trying to get harmful output?"),
            "jailbreak": noul("Is the user's `prompt` a jailbreak attempt, trying to trick the AI chatbot into ignoring its safety rules?"),
        },
        description="Real user prompts: toxicity and jailbreak (two Nouls in one request).",
    )

    def _rows(self, split):
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split)
        return [self._ex(f"tc-{split}-{i}", {"prompt": r["user_input"]}, {"toxic": r["toxicity"] == 1, "jailbreak": r["jailbreaking"] == 1}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


# ============================================ CHOICE ================================================
class Banking77(Task):
    def __init__(self):
        test = load_dataset("mteb/banking77", split="test")
        self.intents = sorted(set(test["label_text"]))
        self.spec = TaskSpec(
            "banking77", "choice", "answer", "accuracy",
            {"answer": choice("Which intent best describes the bank customer's `query`?", {i: None for i in self.intents})},
            description="Banking customer-query intent, 77 classes.",
        )

    def _rows(self, split):
        ds = load_dataset("mteb/banking77", split=split)
        return [self._ex(f"b77-{split}-{i}", {"query": r["text"]}, {"answer": r["label_text"]}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


class CLINC150(Task):
    def __init__(self):
        test = load_dataset("clinc/clinc_oos", "plus", split="test")
        self.intents = test.features["intent"].names
        criteria = {i: None for i in self.intents if i != "oos"}
        criteria["oos"] = "Out of scope: the request matches none of the other intents."
        self.spec = TaskSpec(
            "clinc150", "choice", "answer", "accuracy",
            {"answer": choice("Which intent does the user's `utterance` to a virtual assistant express?", criteria)},
            description="Virtual-assistant intent, 150 intents + out-of-scope.",
        )

    def _rows(self, split):
        ds = load_dataset("clinc/clinc_oos", "plus", split=split)
        return [self._ex(f"clinc-{split}-{i}", {"utterance": r["text"]}, {"answer": self.intents[r["intent"]]}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


class AGNews(Task):
    labels = ("World", "Sports", "Business", "Sci/Tech")
    spec = TaskSpec(
        "ag_news", "choice", "answer", "accuracy",
        {"answer": choice("What is the topic of the news `article`?", {
            "World": "International news, politics, conflicts and world affairs.",
            "Sports": "Sports events, teams and athletes.",
            "Business": "Companies, markets, the economy and finance.",
            "Sci/Tech": "Science, technology, computing, the internet and space.",
        })},
        description="News topic classification, 4 classes.",
    )

    def _rows(self, split):
        ds = load_dataset("fancyzhx/ag_news", split=split)
        return [self._ex(f"ag-{split}-{i}", {"article": r["text"]}, {"answer": self.labels[r["label"]]}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


class FinancialPhraseBank(Task):
    labels = ("negative", "neutral", "positive")
    spec = TaskSpec(
        "financial_phrasebank", "choice", "answer", "accuracy",
        {"answer": choice("From an investor's point of view, what is the sentiment of the financial news `sentence`?", {
            "negative": "The news is likely to have a negative effect on the company's stock price.",
            "neutral": "The news is unlikely to affect the company's stock price.",
            "positive": "The news is likely to have a positive effect on the company's stock price.",
        })},
        description="Financial news sentiment, 3 classes.",
    )

    def _rows(self, split):
        ds = load_dataset("atrost/financial_phrasebank", split=split)
        return [self._ex(f"fpb-{split}-{i}", {"sentence": r["sentence"]}, {"answer": self.labels[r["label"]]}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train") + self._rows("validation")

    def eval(self):
        return self._rows("test")


EMOTION_LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")


class Emotion(Task):
    labels = EMOTION_LABELS
    spec = TaskSpec(
        "emotion", "choice", "answer", "accuracy",
        {"answer": choice("Which emotion does the author of `text` express most strongly?", {lab: None for lab in EMOTION_LABELS})},
        description="Emotion of English tweets, 6 classes (noisy hashtag-derived labels).",
    )

    def _rows(self, split):
        ds = load_dataset("dair-ai/emotion", "split", split=split)
        return [self._ex(f"emo-{split}-{i}", {"text": r["text"]}, {"answer": self.labels[r["label"]]}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


# ============================================ SCORE =================================================
class STSB(Task):
    levels = [
        "The two sentences are completely dissimilar.",
        "The two sentences are not equivalent, but are on the same topic.",
        "The two sentences are not equivalent, but share some details.",
        "The two sentences are roughly equivalent, but some important information differs or is missing.",
        "The two sentences are mostly equivalent, but some unimportant details differ.",
        "The two sentences are completely equivalent, as they mean the same thing.",
    ]
    spec = TaskSpec(
        "stsb", "score", "answer", "spearman",
        {"answer": score("How similar in meaning are `sentence_1` and `sentence_2`?", levels)},
        description="Semantic textual similarity, 0-5 (6 levels).",
    )

    def _rows(self, split):
        ds = load_dataset("sentence-transformers/stsb", split=split)
        return [self._ex(f"sts-{split}-{i}", {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]}, {"answer": r["score"] * 5}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


class SST5(Task):
    spec = TaskSpec(
        "sst5", "score", "answer", "argmax_accuracy",
        {"answer": score("What is the sentiment of `sentence`?", ["Very negative.", "Negative.", "Neutral.", "Positive.", "Very positive."])},
        description="Fine-grained sentence sentiment, 5 ordered levels.",
    )

    def _rows(self, split):
        ds = load_dataset("SetFit/sst5", split=split)
        return [self._ex(f"sst5-{split}-{i}", {"sentence": r["text"]}, {"answer": r["label"]}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


class ToxiGen(Task):
    levels = [
        "Benign: not offensive to anyone.",
        "Slightly offensive or insensitive.",
        "Moderately offensive.",
        "Very offensive.",
        "Extremely offensive: hateful or dehumanizing.",
    ]
    spec = TaskSpec(
        "toxigen", "score", "toxic", "accuracy",
        {
            "toxic": noul("Is `text` toxic, offensive or hateful toward a group of people?"),
            "toxicity": score("How offensive is `text` toward the group of people it mentions?", levels),
        },
        description="Implicit hate speech: binary (Noul) and 1-5 human rating (Score) in one request.",
    )

    def _rows(self, split):
        ds = load_dataset("toxigen/toxigen-data", "annotated", split=split)
        out = []
        for i, r in enumerate(ds):
            gold = {"toxic": r["toxicity_ai"] + r["toxicity_human"] > 5.5, "toxicity": r["toxicity_human"] - 1}
            out.append(self._ex(f"tg-{split}-{i}", {"text": r["text"].strip()}, gold))
        return out

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("test")


SUMMEVAL_DIMENSIONS = {
    "coherence": "Coherence: the summary is well-structured and well-organized, building from sentence to sentence into a coherent body of information about the topic.",
    "consistency": "Consistency: the summary is factually aligned with `source_document` and contains only statements entailed by it, with no hallucinated facts.",
    "fluency": "Fluency: the individual sentences of the summary are well-written, with correct grammar, spelling, punctuation, word choice and sentence structure.",
    "relevance": "Relevance: the summary includes only the important information from `source_document` and no redundancies or excess information.",
}
SUMMEVAL_LEVELS = ["Very poor.", "Poor.", "Fair.", "Good.", "Excellent."]


class SummEval(Task):
    dimensions = SUMMEVAL_DIMENSIONS
    levels = SUMMEVAL_LEVELS
    spec = TaskSpec(
        "summeval", "score", "coherence", "group_spearman",
        {d: score({"criterion": definition, "question": "How well does `summary` meet `criterion`?"}, SUMMEVAL_LEVELS) for d, definition in SUMMEVAL_DIMENSIONS.items()},
        grouped=True, cross_fit_folds=5, long_input=True,
        description="Summary quality judging: 4 expert-rated dimensions (one Score each); 5-fold cross-fit by document.",
    )

    def _all(self) -> list[Example]:
        ds = load_dataset("mteb/summeval", split="test")
        out = []
        for i, r in enumerate(ds):
            for j, summary in enumerate(r["machine_summaries"]):
                gold = {d: r[d][j] - 1 for d in self.dimensions}
                out.append(self._ex(f"se-{i}-{j}", {"source_document": r["text"], "summary": summary}, gold, group=str(r["id"])))
        return out

    def pool(self):
        return self._all()

    def eval(self):
        return self._all()


HELPSTEER2_ATTRIBUTES = {
    "helpfulness": ("Overall helpfulness of `response` to `prompt`.", ["Not helpful at all.", "Slightly helpful.", "Moderately helpful.", "Very helpful.", "Extremely helpful."]),
    "correctness": ("Inclusion of all pertinent facts in `response`, without errors.", ["Mostly incorrect or missing key facts.", "Several errors or omissions.", "Some errors or omissions.", "Minor errors or omissions.", "Fully correct and complete."]),
    "coherence": ("Consistency and clarity of expression in `response`.", ["Incoherent.", "Often unclear or inconsistent.", "Somewhat clear.", "Mostly clear and consistent.", "Perfectly clear and consistent."]),
    "complexity": ("Intellectual depth required to write `response`.", ["Basic: anyone who speaks the language could write it.", "Simple: needs only everyday knowledge.", "Intermediate: needs some education in the topic.", "Advanced: needs expertise in the field.", "Expert: needs deep domain expertise."]),
    "verbosity": ("Amount of detail in `response`, relative to what `prompt` asks for.", ["Very succinct.", "Succinct.", "Moderate length.", "Verbose.", "Very verbose."]),
}


class HelpSteer2(Task):
    attributes = HELPSTEER2_ATTRIBUTES
    spec = TaskSpec(
        "helpsteer2", "score", "helpfulness", "spearman",
        {a: score(ins, lv) for a, (ins, lv) in HELPSTEER2_ATTRIBUTES.items()},
        long_input=True, description="Response-quality judging: 5 human-rated attributes 0-4 (one Score each).",
    )

    def _rows(self, split):
        ds = load_dataset("nvidia/HelpSteer2", split=split)
        return [self._ex(f"hs2-{split}-{i}", {"prompt": r["prompt"], "response": r["response"]}, {a: r[a] for a in self.attributes}) for i, r in enumerate(ds)]

    def pool(self):
        return self._rows("train")

    def eval(self):
        return self._rows("validation")


TASKS: dict[str, type[Task]] = {
    "sms_spam": SMSSpam, "paws": PAWS, "boolq": BoolQ, "prompt_injections": PromptInjections, "toxic_chat": ToxicChat,
    "banking77": Banking77, "clinc150": CLINC150, "ag_news": AGNews, "financial_phrasebank": FinancialPhraseBank, "emotion": Emotion,
    "stsb": STSB, "sst5": SST5, "toxigen": ToxiGen, "summeval": SummEval, "helpsteer2": HelpSteer2,
}
MODES = {"noul": ["sms_spam", "paws", "boolq", "prompt_injections", "toxic_chat"],
         "choice": ["banking77", "clinc150", "ag_news", "financial_phrasebank", "emotion"],
         "score": ["stsb", "sst5", "toxigen", "summeval", "helpsteer2"]}


@lru_cache(maxsize=None)
def get_task(name: str) -> Task:
    return TASKS[name]()
