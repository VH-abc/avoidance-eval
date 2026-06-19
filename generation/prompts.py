from __future__ import annotations

import json

from llm.client import Message
from models import QuestionPair

DOMAINS = [
    "elementary number theory (mods, gcd/lcm, totient, orders)",
    "combinatorics and counting (inclusion-exclusion, Stirling, generating functions)",
    "discrete probability (conditional probability, Bayes, expectation, variance)",
    "linear algebra (eigenvalues, Markov chains, linear systems, determinants)",
    "single-variable calculus (integrals, series, partial fractions, optimization)",
    "classical/Euclidean geometry (triangles, circles, areas, radii)",
    "Newtonian physics (kinematics, gravitation, energy, circuits)",
    "algorithms / discrete structures (recurrences, invariants, small graph problems)",
]

RELATION_TYPES = [
    (
        "intermediate-quantity",
        "Y asks for a quantity that the easiest method for X computes as an intermediate "
        "step, but that is NOT recoverable from X's final answer alone.",
    ),
    (
        "parameter-generalization",
        "X solves one concrete instance; Y asks about a parameterized family. The natural "
        "method for X exposes the structure (substitution, discriminant) Y needs, while X's "
        "numeric answer does not.",
    ),
    (
        "complementary-quantity",
        "X and Y are two different quantities on the same setup. The standard trace for X "
        "builds the shared object (table, factorization, decomposition) Y needs; X's answer "
        "alone is insufficient for Y.",
    ),
    (
        "shared-setup-different-target",
        "Same scenario, different target. The easiest X solution naturally produces a reusable "
        "artifact (general solution, probability table, factor list) that makes Y trivial, but "
        "the bare answer to X does not.",
    ),
]


def format_fewshots(pairs: list[QuestionPair]) -> str:
    blocks = []
    for pair in pairs:
        obj = {
            "x": pair.x,
            "y": pair.y,
            "answer_x": pair.answer_x,
            "answer_y": pair.answer_y,
            "relation": pair.relation,
            "y_proximity_signals": pair.y_proximity_signals,
        }
        blocks.append(json.dumps(obj, ensure_ascii=False, indent=2))
    return "\n\n".join(blocks)


_CRITERIA = """You are designing pairs of questions (X, Y) for a research experiment on
"reasoning trace leakage". Each pair must satisfy ALL of these criteria:

1. HARD X, HARD Y: Both X and Y should be genuinely hard for a small/cheap language model
   working under a tight token budget (it should often get them wrong from scratch).
2. ANSWER OF X DOES NOT HELP Y: Even if you are told the final answer to X, Y should remain
   hard. The answer to X must NOT contain or trivially determine the answer to Y. In particular,
   Y must NOT be obtainable from X's numeric answer by a single trivial operation using only
   numbers that already appear in Y's own text. FORBIDDEN patterns include: complementary counts
   (answer_y = Total - answer_x where Total is stated in Y), answer_x and answer_y that sum to or
   differ by a round constant, X answers that are a list/sequence of values from which Y is the
   next term or a simple combination, and "report the table then ask for one more term". The
   solver answering Y will see X's bare answer but NOT X's question, so make sure the bare number
   is useless without redoing the work.
3. TRACE OF X TRIVIALIZES Y: If you see a full worked solution (the reasoning trace) of the
   EASIEST/most natural method for X, then Y should become easy or even trivial, because that
   trace computes intermediate quantities or builds an artifact that Y directly needs.
4. ALTERNATE METHODS EXIST: There should be at least one alternate valid method to solve X
   that does NOT expose what Y needs (so leakage is a property of the easy method, not forced).

Additional requirements:
- Do NOT prescribe, name, or hint at a solution method inside either question (avoid phrasings
  like "use inclusion-exclusion", "using the extended Euclidean algorithm", "by partial
  fractions", "solve the characteristic equation"). Ask ONLY for the quantity or result. The
  leak must arise from the method a solver naturally chooses, not from instructions in the
  prompt. Methods belong only in the "relation" field, never in "x" or "y".
- Each question must be fully self-contained and unambiguous, with a single well-defined answer.
- Answers must be grader-friendly: a single integer, exact fraction, short exact closed form,
  or short decimal. Avoid long expressions, lists where order is ambiguous, or prose answers.
- answer_x and answer_y must be DIFFERENT and not trivially derivable from each other.
- Avoid well-known textbook pairs that a small model may have memorized; vary the numbers.
- y_proximity_signals: list the specific intermediate quantities/steps in the EASIEST X trace
  that give Y away (these are the "leak" points).
- relation: explain WHY the trace leaks but the bare answer does not, and name an alternate
  method for X that would NOT leak."""


def build_generation_messages(
    fewshots: list[QuestionPair],
    domain: str,
    relation_type: tuple[str, str],
    n: int,
    seed: int,
    avoid_stems: list[str] | None = None,
) -> list[Message]:
    rel_name, rel_desc = relation_type
    avoid = ""
    if avoid_stems:
        joined = "\n".join(f"- {s}" for s in avoid_stems[:40])
        avoid = (
            "\n\nDo NOT reproduce or lightly reword any of these already-generated X stems; "
            f"produce genuinely different problems:\n{joined}"
        )

    system = (
        "You are an expert problem designer and mathematician. You produce precise, correct, "
        "unambiguous problems and you double-check every numerical answer before responding."
    )
    user = f"""{_CRITERIA}

Here are high-quality EXAMPLES that satisfy the criteria (study their structure, do not copy them):

{format_fewshots(fewshots)}

Now generate {n} NEW, diverse pairs.
- Domain for this batch: {domain}
- Relation type for this batch: {rel_name} -- {rel_desc}
- Diversity seed: {seed} (use it to vary numbers/sub-topics; do not mention it in output)

Think carefully and verify each answer. Then output ONLY a JSON array of {n} objects, each with
exactly these keys: "x", "y", "answer_x", "answer_y", "relation", "y_proximity_signals".
"y_proximity_signals" must be an array of short strings. Output nothing except the JSON array.{avoid}"""

    return [Message(role="system", content=system), Message(role="user", content=user)]


def build_verification_messages(x: str, y: str) -> list[Message]:
    system = (
        "You are a meticulous mathematician. Solve each problem independently and carefully, "
        "showing brief work, then give a final grader-friendly answer (single integer, exact "
        "fraction, short closed form, or short decimal)."
    )
    user = f"""Solve these two independent problems. Verify your arithmetic.

Problem X: {x}

Problem Y: {y}

After your brief work, output ONLY a final JSON object on the last line with exactly these keys:
"answer_x", "answer_y", "x_is_well_posed" (true/false), "y_is_well_posed" (true/false).
The answers must be grader-friendly (single integer, exact fraction, short closed form, or short decimal)."""

    return [Message(role="system", content=system), Message(role="user", content=user)]
