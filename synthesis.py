#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates three JSON files—test.json, train.json, valid.json—
each containing a randomized mix of reflexive, symmetric,
and transitive (equality-chain) reasoning examples.

Usage:
    python make_combined_datasets.py
"""

import json
import random
import string
from pathlib import Path
from typing import List, Set

# ─── GLOBALS ──────────────────────────────────────────────────────────────────
RAND_SEED = 42
random.seed(RAND_SEED)

POOL_SIZE  = 50_000
NAME_RANGE = (4, 7)
_names: Set[str] = set()
while len(_names) < POOL_SIZE:
    L = random.randint(*NAME_RANGE)
    _names.add("".join(random.choices(string.ascii_uppercase, k=L)))
POOL_OF_NAMES: List[str] = list(_names)
del _names

def pick_unused(exclude: Set[str]) -> str:
    """Return a random name not in `exclude`."""
    while True:
        name = random.choice(POOL_OF_NAMES)
        if name not in exclude:
            return name

# ─── BUILDERS ─────────────────────────────────────────────────────────────────

def build_symmetric_example(pos: bool, ex_id: str) -> dict:
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
    QUES = [
        "Does {b} equal {a}?",
        "Is it true that {b} is the same as {a}?",
        "Can we say {b} and {a} refer to the same thing?"
    ]
    a, b    = random.sample(POOL_OF_NAMES, 2)
    premise = random.choice(T_POS if pos else T_NEG).format(a=a, b=b)
    question= random.choice(QUES).format(a=a, b=b)
    return {"id": ex_id,
            "text": f"{premise} {question}",
            "label": "yes" if pos else "no"}

def build_reflexive_example(pos: bool, ex_id: str) -> dict:
    T_POS = [
        "{a} equals {b}.",
        "{a} is the same as {b}.",
        "{a} and {b} refer to the same object."
    ]
    T_NEG = [
        "{a} teaches {b}.",
        "{a} is the boss of {b}.",
        "{a} is taller than {b}."
    ]
    Q_POS = [
        "Does {a} equal {a}?",
        "Is it true that {a} is the same as itself?",
        "Can we say {a} refers to itself?"
    ]
    Q_NEG = [
        "Does {a} teach {a}?",
        "Is {a} the boss of {a}?",
        "Is {a} taller than {a}?"
    ]
    a, b    = random.sample(POOL_OF_NAMES, 2)
    premise = random.choice(T_POS if pos else T_NEG).format(a=a, b=b)
    question= random.choice(Q_POS if pos else Q_NEG).format(a=a)
    return {"id": ex_id,
            "text": f"{premise} {question}",
            "label": "yes" if pos else "no"}

def build_eq_chain_example(L: int, pos: bool, ex_id: str,
                           distract_p=0.4, distract_max=3) -> dict:
    P_TEMPL = [
        "{a} equals {b}.",
        "{a} is the same as {b}.",
        "It is established that {a} = {b}.",
        "{a} and {b} refer to one object."
    ]
    Q_TEMPL = [
        "Therefore, does {a} equal {c}?",
        "Given this, is {a} the same as {c}?",
        "Hence, can we say {a} = {c}?"
    ]
    DISTRACT = [
        "{a} equals {b}.",
        "{a} is identical to {b}."
    ]

    chain = random.sample(POOL_OF_NAMES, k=L)
    premises = [
        random.choice(P_TEMPL).format(a=chain[i], b=chain[i+1])
        for i in range(L-1)
    ]
    a = chain[0]
    c = chain[-1] if pos else pick_unused(set(chain))
    label = "yes" if pos else "no"
    question = random.choice(Q_TEMPL).format(a=a, c=c)

    if random.random() < distract_p:
        for _ in range(random.randint(1, distract_max)):
            x, y = random.sample(POOL_OF_NAMES, 2)
            premises.insert(
                random.randrange(len(premises)+1),
                random.choice(DISTRACT).format(a=x, b=y)
            )

    random.shuffle(premises)
    return {
        "id": ex_id,
        "text": " ".join(premises) + " " + question,
        "label": label,
        "length": L
    }

# ─── GENERATORS ────────────────────────────────────────────────────────────────

def generate_symmetric(n: int, p_pos: float, stem: str) -> List[dict]:
    return [
        build_symmetric_example(random.random() < p_pos, f"{stem}-{i:06d}")
        for i in range(n)
    ]

def generate_reflexive(n: int, p_pos: float, stem: str) -> List[dict]:
    return [
        build_reflexive_example(random.random() < p_pos, f"{stem}-{i:06d}")
        for i in range(n)
    ]

def generate_eq_chain(n: int, p_pos: float, stem: str,
                      L_choices=range(2, 4)) -> List[dict]:
    return [
        build_eq_chain_example(
            L=random.choice(L_choices),
            pos=(random.random() < p_pos),
            ex_id=f"{stem}-{i:06d}"
        )
        for i in range(n)
    ]

# ─── QUANTIFIER EXAMPLES ────────────────────────────────────────────────────────

def build_quantifier_example(pos: bool, ex_id: str) -> dict:
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
        "no": {
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

    A, B, C = random.sample(POOL_OF_NAMES, 3)
    while True:
        q1 = random.choice(QUANTS)
        q2 = random.choice(QUANTS)
        entailed = ENTAILS.get((q1, q2), set())
        if pos and entailed:
            q_query = random.choice(sorted(entailed))
            label = "yes"
            break
        if not pos:
            missing = [q for q in QUANTS if q not in entailed]
            if missing:
                q_query = random.choice(missing)
                label = "no"
                break

    prem1 = random.choice(quant_templates[q1]["premises"]).format(a=A, b=B)
    prem2 = random.choice(quant_templates[q2]["premises"]).format(a=B, b=C)
    question = random.choice(quant_templates[q_query]["questions"]).format(a=A, c=C)

    if random.random() < 0.5:
        text = f"{prem1} {prem2} {question}"
    else:
        text = f"{prem2} {prem1} {question}"

    return {"id": ex_id, "text": text, "label": label}

def generate_quantifier(n: int, p_pos: float, stem: str) -> List[dict]:
    return [
        build_quantifier_example(random.random() < p_pos, f"{stem}-{i:06d}")
        for i in range(n)
    ]

# ─── NEGATION EXAMPLES ─────────────────────────────────────────────────────────

def build_negation_example(L: int, pos: bool, ex_id: str) -> dict:
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

    chain = random.sample(POOL_OF_NAMES, k=L)
    premises = [
        random.choice(EQ).format(a=chain[i], b=chain[i+1])
        for i in range(L - 1)
    ]
    if not pos:
        idx = random.randint(0, L - 2)
        premises[idx] = random.choice(NEQ).format(a=chain[idx], b=chain[idx + 1])
    random.shuffle(premises)

    a, c = chain[0], chain[-1]
    question = random.choice(Q).format(a=a, c=c)
    return {"id": ex_id, "text": " ".join(premises) + " " + question, "label": "yes" if pos else "no"}

def generate_negation(n: int, p_pos: float, stem: str, L_choices=range(3, 7)) -> List[dict]:
    return [
        build_negation_example(random.choice(L_choices), random.random() < p_pos, f"{stem}-{i:06d}")
        for i in range(n)
    ]

# ─── RELATIONS EXAMPLES ────────────────────────────────────────────────────────

def build_relations_example(L: int, rel: str, pos: bool, ex_id: str) -> dict:
    REL = {
        "brother": {
            "trans": True,
            "p": ["{a} is {b}'s brother.", "{a} and {b} are brothers."],
            "q": ["Is {a} {c}'s brother?", "Can we say that {a} and {c} are brothers?"],
        },
        "father": {
            "trans": False,
            "p": ["{a} is {b}'s father.", "{b} is the child of {a}."],
            "q": ["Is {a} {c}'s father?", "Can we say {a} is an ancestor of {c}?"],
        },
        "friend": {
            "trans": False,
            "p": ["{a} is friends with {b}.", "{a} and {b} are friends."],
            "q": ["Are {a} and {c} friends?", "Can we say {a} is friends with {c}?"],
        },
        "coworker": {
            "trans": True,
            "p": ["{a} is a coworker of {b}.", "{a} and {b} work together."],
            "q": ["Is {a} a coworker of {c}?", "Do {a} and {c} work together?"],
        },
    }

    chain = random.sample(POOL_OF_NAMES, k=L)
    premises = [
        random.choice(REL[rel]["p"]).format(a=chain[i], b=chain[i + 1])
        for i in range(L - 1)
    ]
    a, c = chain[0], chain[-1]

    if REL[rel]["trans"]:
        label = "yes" if pos else "no"
        if not pos:
            c = pick_unused(set(chain))
    else:
        label = "no" if pos else "yes"

    question = random.choice(REL[rel]["q"]).format(a=a, c=c)
    random.shuffle(premises)
    return {"id": ex_id, "text": " ".join(premises) + " " + question, "label": label}

def generate_relations(n: int, p_pos: float, stem: str, L_choices=range(2, 7)) -> List[dict]:
    rels = ["brother", "father", "friend", "coworker"]
    return [
        build_relations_example(random.choice(L_choices),
                                random.choice(rels),
                                random.random() < p_pos,
                                f"{stem}-{i:06d}")
        for i in range(n)
    ]

# ─── OUTPUT ──────────────────────────────────────────────────────────────────

def write_json(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] {len(data):5d} examples → {path}")

def make_all_combined(out_dir: Path,
                      n_train=5000, n_valid=500, n_test=500,
                      p_pos_train=0.7, p_pos_test=0.5) -> None:
    # create each split
    train = (
        generate_symmetric(n_train, p_pos_train, "sym-tr") +
        generate_reflexive(n_train, p_pos_train, "refl-tr") +
        generate_eq_chain(n_train, p_pos_train, "eq-tr") +
        generate_quantifier(n_train, p_pos_train, "quant-tr") +
        generate_negation(n_train, p_pos_train, "neg-tr") +
        generate_relations(n_train, p_pos_train, "rel-tr")
    )
    valid = (
        generate_symmetric(n_valid, p_pos_train, "sym-va") +
        generate_reflexive(n_valid, p_pos_train, "refl-va") +
        generate_eq_chain(n_valid, p_pos_train, "eq-va") +
        generate_quantifier(n_valid, p_pos_train, "quant-va") +
        generate_negation(n_valid, p_pos_train, "neg-va") +
        generate_relations(n_valid, p_pos_train, "rel-va")
    )
    test  = (
        generate_symmetric(n_test,  p_pos_test,  "sym-te") +
        generate_reflexive(n_test,  p_pos_test,  "refl-te") +
        generate_eq_chain(n_test,   p_pos_test,  "eq-te") +
        generate_quantifier(n_test,  p_pos_test, "quant-te") +
        generate_negation(n_test, p_pos_test,    "neg-te") +
        generate_relations(n_test, p_pos_test,   "rel-te")
    )

    # shuffle each to mix families
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    # write in the order: test, train, valid
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "test.json",  test)
    write_json(out_dir / "train.json", train)
    write_json(out_dir / "valid.json", valid)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_all_combined(Path("datasets"))