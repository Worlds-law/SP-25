#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
─────────────────────────
Produces natural-language reasoning datasets that test
• Reflexivity
• Symmetry
• Transitivity

Usage
-----
python make_datasets.py           # generate everything
python make_datasets.py refl      # only reflexive
python make_datasets.py sym       # only symmetric
python make_datasets.py trans     # only transitive
"""


# GLOBAL IMPORTS & UTILITIES
import json, random, string, os, sys
from pathlib import Path
from typing import List, Tuple

RAND_SEED = 42
random.seed(RAND_SEED)

# Shared pool of arbitrary uppercase names
POOL_SIZE   = 50_000
NAME_RANGE  = (4, 7)                       # inclusive length in letters
_names: set[str] = set()
while len(_names) < POOL_SIZE:
    L = random.randint(*NAME_RANGE)
    _names.add("".join(random.choices(string.ascii_uppercase, k=L)))
POOL_OF_NAMES: List[str] = list(_names)
del _names                                                     # keep memory clean


def pick_unused(exclude: set[str]) -> str:
    "Return a random entity name not in *exclude*."
    while True:
        name = random.choice(POOL_OF_NAMES)
        if name not in exclude:
            return name


def write_jsonl(path: Path, objs: List[dict]) -> None:
    """Write one JSON object per line (utf-8, non-ASCII allowed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in objs:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[✓] {len(objs):5d} examples  → {path}")



# SYMMETRIC

def make_symmetric_splits(out_root: Path,
                          n_train=5_000, n_valid=500,
                          n_test_iid=500, n_test_long=500,
                          p_pos_train=0.70, p_pos_test=0.50) -> None:
    """
    Generates datasets where the premise ‘A = B’ (or similar)
    is queried as ‘Does B = A?’
    """
    T_POS = [
        "{a} equals {b}.",
        "{a} is the same as {b}.",
        "{a} and {b} refer to one object."
    ]
    T_NEG = [
        "{a} teaches {b}.",
        "{a} is the boss of {b}.",
        "{a} is taller than {b}."
    ]
    QUESTIONS = [
        "Does {b} equal {a}?",
        "Is it true that {b} is the same as {a}?",
        "Can we say {b} and {a} refer to the same thing?"
    ]

    def example(positive: bool, ex_id: str) -> dict:
        a, b = random.sample(POOL_OF_NAMES, 2)
        premise = random.choice(T_POS if positive else T_NEG).format(a=a, b=b)
        question = random.choice(QUESTIONS).format(a=a, b=b)
        label = "yes" if positive else "no"
        return {"id": ex_id, "text": f"{premise} {question}", "label": label}

    def split(count, p_pos, stem) -> List[dict]:
        return [example(random.random() < p_pos, f"{stem}-{i:06d}")
                for i in range(count)]

    base = out_root / "symmetric"
    write_jsonl(base / "sym_train.jsonl",      split(n_train,      p_pos_train, "sym-tr"))
    write_jsonl(base / "sym_valid.jsonl",      split(n_valid,      p_pos_train, "sym-va"))
    write_jsonl(base / "sym_test_iid.jsonl",   split(n_test_iid,   p_pos_test,  "sym-ti"))
    write_jsonl(base / "sym_test_long.jsonl",  split(n_test_long,  p_pos_test,  "sym-tl"))



# REFLEXIVE  (self-relation)

def make_reflexive_splits(out_root: Path,
                          n_train=5_000, n_valid=500,
                          n_test_iid=500, n_test_long=500,
                          p_pos_train=0.70, p_pos_test=0.50) -> None:
    """
    True positives:  premises ‘A = B’  →  question ‘Does A = A?’
    Negatives use non-reflexive verbs (teaches, taller than …).
    """
    T_POS = [
        "{a} equals {b}.",
        "{a} is the same as {b}.",
        "{a} and {b} refer to the same object.",
    ]
    T_NEG = [
        "{a} teaches {b}.",
        "{a} is the boss of {b}.",
        "{a} is taller than {b}.",
    ]
    Q_POS = [
        "Does {a} equal {a}?",
        "Is it true that {a} is the same as itself?",
        "Can we say {a} refers to itself?",
    ]
    Q_NEG = [
        "Does {a} teach {a}?",
        "Is {a} the boss of {a}?",
        "Is {a} taller than {a}?",
    ]

    def example(positive: bool, ex_id: str) -> dict:
        a, b = random.sample(POOL_OF_NAMES, 2)
        premise  = random.choice(T_POS if positive else T_NEG).format(a=a, b=b)
        question = random.choice(Q_POS if positive else Q_NEG).format(a=a)
        label = "yes" if positive else "no"
        return {"id": ex_id, "text": f"{premise} {question}", "label": label}

    def split(count, p_pos, stem):
        return [example(random.random() < p_pos, f"{stem}-{i:06d}")
                for i in range(count)]

    base = out_root / "reflexive"
    write_jsonl(base / "refl_train.jsonl",      split(n_train,      p_pos_train, "refl-tr"))
    write_jsonl(base / "refl_valid.jsonl",      split(n_valid,      p_pos_train, "refl-va"))
    write_jsonl(base / "refl_test_iid.jsonl",   split(n_test_iid,   p_pos_test,  "refl-ti"))
    write_jsonl(base / "refl_test_long.jsonl",  split(n_test_long,  p_pos_test,  "refl-tl"))



# TRANSITIVE  (several generators)
# All generators share the same out_root/'transitive' base.
TRANS_BASE = Path("datasets") / "transitive"


_TRANS_TYPES = {}
def register_trans(func):
    _TRANS_TYPES[func.__name__] = func
    return func


# A  Quantifier reasoning (“all”, “some”, “no”)
@register_trans
def trans_quantifier(n_train=20_000, n_valid=3_000, n_test=3_000,
                     positive_ratio=0.5) -> None:
    quant_templates = {
        "all": {
            "premises": [
                "All {a} are {b}.", "Every {a} is a {b}.", "{b}s encompass all {a}."
            ],
            "questions": [
                "Therefore, are all {a} {c}?",
                "Given this, does every {a} count as a {c}?",
            ],
        },
        "some": {
            "premises": [
                "Some {a} are {b}.", "At least one {a} is a {b}.",
                "There exist {a} that are {b}.",
            ],
            "questions": [
                "Therefore, are some {a} {c}?",
                "Given this, can we say at least one {a} is a {c}?",
            ],
        },
        "no":   {
            "premises": [
                "No {a} are {b}.", "Not a single {a} is a {b}.", "{a} are never {b}.",
            ],
            "questions": [
                "Therefore, are no {a} {c}?",
                "Given this, can we conclude that not a single {a} is a {c}?",
            ],
        },
    }
    ENTAILS = {
        ("all",  "all"):  {"all", "some"},
        ("all",  "some"): {"some"},
        ("some", "all"):  {"some"},
        ("all",  "no"):   {"no"},
        ("no",   "all"):  {"no"},
        ("no",   "some"): {"no"},
    }
    QUANTS = list(quant_templates.keys())

    def make_example(positive=True, ex_id="quant-0"):
        A, B, C = random.sample(POOL_OF_NAMES, k=3)
        while True:
            q1, q2 = random.choice(QUANTS), random.choice(QUANTS)
            entailed = ENTAILS.get((q1, q2), set())
            if positive and entailed:
                q_query = random.choice(sorted(entailed))
                label = "yes"
                break
            if (not positive) and (missing := [q for q in QUANTS if q not in entailed]):
                q_query = random.choice(missing)
                label = "no"
                break
        prem1 = random.choice(quant_templates[q1]["premises"]).format(a=A, b=B)
        prem2 = random.choice(quant_templates[q2]["premises"]).format(a=B, b=C)
        question = random.choice(quant_templates[q_query]["questions"]).format(a=A, c=C)
        random.shuffle([prem1, prem2])
        return {
            "id": ex_id, "text": f"{prem1} {prem2} {question}", "label": label,
            "link_q1": q1, "link_q2": q2, "query_q": q_query,
        }

    def split(count, stem):
        return [make_example(random.random()<positive_ratio, f"{stem}-{i:06d}")
                for i in range(count)]

    folder = TRANS_BASE / "quantifiers"
    write_jsonl(folder / "trans_quant_train.jsonl", split(n_train, "q-tr"))
    write_jsonl(folder / "trans_quant_valid.jsonl", split(n_valid, "q-va"))
    write_jsonl(folder / "trans_quant_test.jsonl",  split(n_test,  "q-te"))


# B  Equality-chain transitivity (with distractors)
@register_trans
def trans_eq_chain(n_train=5_000, n_valid=500,
                   n_test_iid=500, n_test_long=500,
                   p_pos_train=0.70, p_pos_test=0.50,
                   distract_p=0.4, distract_max=3) -> None:
    P_TEMPL = [
        "{a} equals {b}.",
        "{a} is the same as {b}.",
        "It is established that {a} = {b}.",
        "{a} and {b} refer to one object.",
    ]
    Q_TEMPL = [
        "Therefore, does {a} equal {c}?",
        "Given this, is {a} the same as {c}?",
        "Hence, can we say {a} = {c}?",
    ]
    DISTRACT = [
        "{a} equals {b}.",
        "{a} is identical to {b}.",
    ]

    def build(L, positive, ex_id):
        chain = random.sample(POOL_OF_NAMES, k=L)
        prem = [random.choice(P_TEMPL).format(a=chain[i], b=chain[i+1])
                for i in range(L-1)]
        a = chain[0]
        c = chain[-1] if positive else pick_unused(set(chain))
        label = "yes" if positive else "no"
        q = random.choice(Q_TEMPL).format(a=a, c=c)
        if random.random() < distract_p:
            for _ in range(random.randint(1, distract_max)):
                x, y = random.sample(POOL_OF_NAMES, 2)
                prem.insert(random.randrange(len(prem)+1),
                            random.choice(DISTRACT).format(a=x, b=y))
        random.shuffle(prem)
        return {"id": ex_id, "text": " ".join(prem) + " " + q,
                "label": label, "length": L}

    def split(n, L_range, p_pos, stem):
        return [build(random.choice(L_range), random.random()<p_pos, f"{stem}-{i:06d}")
                for i in range(n)]

    folder = TRANS_BASE / "eq_chain"
    write_jsonl(folder/"trans_eq_train.jsonl",
                split(n_train, range(2,4), p_pos_train, "eq-tr"))
    write_jsonl(folder/"trans_eq_valid.jsonl",
                split(n_valid, range(2,4), p_pos_train, "eq-va"))
    write_jsonl(folder/"trans_eq_test_iid.jsonl",
                split(n_test_iid, range(2,4), p_pos_test, "eq-ti"))
    write_jsonl(folder/"trans_eq_test_long.jsonl",
                split(n_test_long, range(4,8), p_pos_test, "eq-tl"))


# C  Negated-link chains (“does not equal …”)
@register_trans
def trans_negation(n_train=15_000, n_valid=2_000, n_test=2_000,
                   positive_ratio=0.5, L_range=range(3,7)) -> None:
    EQ = [
        "{a} equals {b}.", "{a} is the same as {b}.",
        "It is established that {a} = {b}.", "{a} and {b} refer to one object.",
    ]
    NEQ = [
        "{a} does not equal {b}.", "{a} is not the same as {b}.",
        "It is false that {a} = {b}.", "{a} and {b} are not equal.",
    ]
    Q = [
        "Therefore, does {a} equal {c}?",
        "Given this, is {a} the same as {c}?",
        "Hence, can we say {a} = {c}?",
    ]

    def build(L, positive, ex_id):
        chain = random.sample(POOL_OF_NAMES, k=L)
        premises = [random.choice(EQ).format(a=chain[i], b=chain[i+1])
                    for i in range(L-1)]
        label = "yes" if positive else "no"
        if not positive:                               # break one random link
            idx = random.randint(0, L-2)
            premises[idx] = random.choice(NEQ).format(a=chain[idx],
                                                      b=chain[idx+1])
        random.shuffle(premises)
        a, c = chain[0], chain[-1]
        query = random.choice(Q).format(a=a, c=c)
        return {"id": ex_id, "text": " ".join(premises)+" "+query,
                "label": label, "length": L, "type": "negation"}

    def split(n, stem):
        return [build(random.choice(L_range), random.random()<positive_ratio,
                      f"{stem}-{i:06d}") for i in range(n)]

    folder = TRANS_BASE / "negation"
    write_jsonl(folder/"trans_neg_train.jsonl",  split(n_train, "neg-tr"))
    write_jsonl(folder/"trans_neg_valid.jsonl",  split(n_valid, "neg-va"))
    write_jsonl(folder/"trans_neg_test.jsonl",   split(n_test,  "neg-te"))


# D  Relation-template chains (brother, father, …)
@register_trans
def trans_relations(n_train=20_000, n_valid=3_000, n_test=3_000,
                    positive_ratio=0.5, L_range=range(2,7)) -> None:
    REL = {
        "brother": {
            "trans": True,
            "p": ["{a} is {b}'s brother.", "{a} and {b} are brothers."],
            "q": ["Is {a} {c}'s brother?",
                  "Can we say that {a} and {c} are brothers?"],
        },
        "father":  {
            "trans": False,
            "p": ["{a} is {b}'s father.", "{b} is the child of {a}."],
            "q": ["Is {a} {c}'s father?",
                  "Can we say {a} is an ancestor of {c}?"],
        },
        "friend":  {
            "trans": False,
            "p": ["{a} is friends with {b}.", "{a} and {b} are friends."],
            "q": ["Are {a} and {c} friends?",
                  "Can we say {a} is friends with {c}?"],
        },
        "coworker":{
            "trans": True,
            "p": ["{a} is a coworker of {b}.", "{a} and {b} work together."],
            "q": ["Is {a} a coworker of {c}?",
                  "Do {a} and {c} work together?"],
        },
    }

    def build(L, relation, positive, ex_id):
        S = REL[relation]
        chain = random.sample(POOL_OF_NAMES, k=L)
        premises = [random.choice(S["p"]).format(a=chain[i], b=chain[i+1])
                    for i in range(L-1)]
        a, c = chain[0], chain[-1]
        if S["trans"]:
            label = "yes" if positive else "no"
            if not positive:
                c = pick_unused(set(chain))
        else:                         # non-transitive relations: invert labels
            label = "no" if positive else "yes"
        query = random.choice(S["q"]).format(a=a, c=c)
        random.shuffle(premises)
        return {"id": ex_id, "text": " ".join(premises)+" "+query,
                "label": label, "length": L, "relation": relation}

    def split(n, stem):
        objs = []
        for i in range(n):
            rel = random.choice(list(REL))
            L = random.choice(L_range)
            objs.append(build(L, rel, random.random()<positive_ratio,
                              f"{stem}-{i:06d}"))
        return objs

    folder = TRANS_BASE / "relations"
    write_jsonl(folder/"trans_rel_train.jsonl", split(n_train, "rel-tr"))
    write_jsonl(folder/"trans_rel_valid.jsonl", split(n_valid, "rel-va"))
    write_jsonl(folder/"trans_rel_test.jsonl",  split(n_test,  "rel-te"))


# E  Mixed kitchen-sink dataset (location, subset, …)
@register_trans
def trans_mixed(n_train=50_000, n_valid=5_000, n_test=5_000,
                positive_ratio=0.7, distract_p=0.4, distract_max=3,
                L_range=range(3,6)) -> None:
    TEMPL = {
        "location": {
            "p": [
                "{a} is located within {b}.", "{a} is situated inside {b}.",
                "{a} is geographically contained in {b}.",
                "{a} resides in {b}.", "{a} is a city in {b}."
            ],
            "q": [
                "Is {a} located within {c}?", "Is {a} situated inside {c}?",
                "Is {a} geographically inside {c}?", "Does {c} contain {a}?",
                "Is {a} a city in {c}?"
            ],
        },
        "neighbour": {
            "p": [
                "{a} is a next-door neighbour of {b}.", "{a} lives next to {b}.",
                "{a} and {b} are next-door neighbours."
            ],
            "q": [
                "Are {a} and {c} next-door neighbours?",
                "Does {a} live next to {c}?",
                "Is {a} a next-door neighbour of {c}?"
            ],
        },
        "subset": {
            "p": [
                "{a} is a subset of {b}.", "All elements of {a} are contained in {b}.",
                "{a} ⊆ {b}.",
            ],
            "q": [
                "Is {a} a subset of {c}?", "Are all elements of {a} contained in {c}?",
                "Does {a} belong wholly to {c}?",
            ],
        },
    }
    DISTRACT = [
        "{a} plays tennis with {b}.",
        "{a} once traveled with {b}.",
        "{a} likes the same music as {b}.",
    ]

    def build(cat, L, positive, ex_id):
        if cat == "neighbour":             # defined as non-transitive here
            positive = False
        chain = random.sample(POOL_OF_NAMES, k=L)
        premises = [random.choice(TEMPL[cat]["p"]).format(a=chain[i], b=chain[i+1])
                    for i in range(L-1)]
        a, c = chain[0], chain[-1]
        label = "yes" if positive else "no"
        query = random.choice(TEMPL[cat]["q"]).format(a=a, c=c)
        # distractors
        if random.random() < distract_p:
            for _ in range(random.randint(1, distract_max)):
                x, y = random.sample(POOL_OF_NAMES, 2)
                premises.insert(random.randrange(len(premises)+1),
                                random.choice(DISTRACT).format(a=x, b=y))
        random.shuffle(premises)
        return {"id": ex_id, "text": " ".join(premises)+" "+query,
                "label": label, "length": L, "category": cat}

    def split(n, stem):
        objs=[]
        for i in range(n):
            cat  = random.choice(list(TEMPL))
            L    = 3 if cat=="divisibility" else random.choice(L_range)
            objs.append(build(cat, L, random.random()<positive_ratio,
                              f"{stem}-{i:06d}"))
        return objs

    folder = TRANS_BASE / "mixed"
    write_jsonl(folder/"mixed_train.jsonl", split(n_train, "mix-tr"))
    write_jsonl(folder/"mixed_valid.jsonl", split(n_valid, "mix-va"))
    write_jsonl(folder/"mixed_test.jsonl",  split(n_test,  "mix-te"))



# DRIVER
def main(which: Tuple[str, ...]) -> None:
    OUT = Path("datasets")                    # root output folder

    if "sym" in which or "all" in which:
        make_symmetric_splits(OUT)
    if "refl" in which or "all" in which:
        make_reflexive_splits(OUT)
    if "trans" in which or "all" in which:
        for func in _TRANS_TYPES.values():
            func()

    print("✓ All requested dataset families are ready.")


if __name__ == "__main__":
    arg = sys.argv[1:]            # allowed: sym, refl, trans
    tokens = {"sym", "refl", "trans"}
    chosen = tuple(tok for tok in arg if tok in tokens) or ("all",)
    main(chosen)