# ──────────────────────────────────────────────────────────────
#  Positive instances: a reflexive relation that is always true
#                      of identical arguments (e.g. “equals”).
#  Negative instances: a non-reflexive relation that does NOT
#                      follow for identical arguments (e.g.
#                      “teaches”, “is taller than”, …).
import json
import random
import string
import os
from pathlib import Path

# ── Global configuration ──────────────────────────────────────
rand_seed                    = 42
random.seed(rand_seed)

pool_size                    = 50_000          # distinct entity names
name_range                   = (4, 7)          # length of each name

num_train                    = 5_000
num_valid                    =   500
num_test_iid                 =   500
num_test_long                =   500           # kept for parity

positive_ratio_for_training  = 0.70
positive_ratio_for_testing   = 0.50            # iid + long

desktop_path   = os.path.join(os.path.expanduser("~"), "Desktop")
out_directory = Path("else/refl_data")
out_directory.mkdir(parents=True, exist_ok=True)

# ── Natural-language templates ────────────────────────────────

templates_positive = [
    "{a} equals {b}.",
    "{a} is the same as {b}.",
    "{a} and {b} refer to the same object.",
]

templates_negative = [
    "{a} teaches {b}.",
    "{a} is the boss of {b}.",
    "{a} is taller than {b}.",
]

questions_positive = [
    "Does {a} equal {a}?",
    "Is it true that {a} is the same as itself?",
    "Can we say {a} refers to itself?",
]

questions_negative = [
    "Does {a} teach {a}?",
    "Is {a} the boss of {a}?",
    "Is {a} taller than {a}?",
]

# ── Name pool ─────────────────────────────────────────────────
names = set()
while len(names) < pool_size:
    L = random.randint(*name_range)
    names.add("".join(random.choices(string.ascii_uppercase, k=L)))
pool_of_names = list(names)

# ── Example factory ───────────────────────────────────────────
def make_example(positive=True, ex_id="reflex-0"):
    a, b = random.sample(pool_of_names, 2)      # b ≠ a

    if positive:
        premise  = random.choice(templates_positive).format(a=a, b=b)
        question = random.choice(questions_positive).format(a=a)
        label    = "yes"
    else:
        premise  = random.choice(templates_negative).format(a=a, b=b)
        question = random.choice(questions_negative).format(a=a)
        label    = "no"

    full_text = f"{premise} {question}"
    return {"id": ex_id, "text": full_text, "label": label}

# ── Split writer ──────────────────────────────────────────────
def write_split(filename: Path, n_examples: int, pos_ratio: float):
    with filename.open("w", encoding="utf-8") as fh:
        for idx in range(n_examples):
            positive = random.random() < pos_ratio
            ex_id    = f"{filename.stem}-{idx:06d}"
            ex       = make_example(positive=positive, ex_id=ex_id)
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {n_examples:5d} → {filename}")

# ── Generate all splits ───────────────────────────────────────
write_split(out_directory / "refl_train.jsonl",       num_train,      positive_ratio_for_training)
write_split(out_directory / "refl_valid.jsonl",       num_valid,      positive_ratio_for_training)
write_split(out_directory / "refl_test_iid.jsonl",    num_test_iid,   positive_ratio_for_testing)
write_split(out_directory / "refl_test_long.jsonl",   num_test_long,  positive_ratio_for_testing)

print("Reflexivity datasets have been generated!")