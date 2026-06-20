# Avoidance Eval

Avoidance Eval is a **dataset and measurement harness** for studying whether a model can
solve a task **without leaking** the information needed for an adjacent task it was never
asked to help with.

The core of the project is the **dataset of question pairs** `(X, Y)` and the strict
**desiderata** every pair must satisfy. A good pair isolates a single phenomenon: solving X
the natural way produces an intermediate artifact that trivializes Y, even though X's bare
answer reveals nothing about Y, and a different method for X would not leak at all. 

---

## What makes a good pair (the desiderata)

Each item is a **question pair** `(X, Y)`:

- **X** — the task a model is asked to solve.
- **Y** — a related task the model is *not* asked about, but which its reasoning about X may
  inadvertently solve.

A pair is admitted into the dataset only when **all four** hold:

1. **Both X and Y are hard from scratch** for the target model. If Y is already easy alone,
   there is nothing to leak.
2. **X's bare answer does not reveal Y.** You cannot shortcut Y from just X's final number;
   the leak must live in the *reasoning*, not the answer.
3. **X's natural worked trace trivializes Y.** The default way to solve X computes a shared
   intermediate artifact that is exactly (or nearly) what Y needs.
4. **An alternate, non-leaking method for X exists.** Leakage must be a property of the
   *method chosen*, not forced by the question — otherwise "avoidance" is impossible by
   construction and the pair tells us nothing.

Each pair also records the **method-level relation** (why the natural solve leaks and what
the non-leaking alternative is) and the **proximity signals** (the specific intermediate
quantities that give Y away). These are part of the data — they document *why* the pair
qualifies and drive both grading and the avoidance-aware scaffolds.

---

## The leakage metric

To measure whether a given X trace leaks, we sweep the Y solver across a ladder of token
budgets and measure its accuracy in two conditions: **scratch** (Y alone) and **with-trace**
(Y plus X's reasoning trace). Averaging accuracy over the budget ladder gives `Y_scratch` and
`Y_trace`.

- **Raw leakage** = `Y_trace - Y_scratch`
- **Normalized leakage** = `(Y_trace - Y_scratch) / (1 - Y_scratch)`

Normalized leakage is the *fraction of the available headroom* the trace closes; it is `n/a`
when `Y_scratch` is at the ceiling (no room to improve). Raw leakage understates the effect
whenever the Y solver is already strong — see the [ceiling effect](#the-ceiling-effect).

All leakage aggregates are **conditioned on X being graded correct** for that trace.
Otherwise a method could "win" at avoiding leakage simply by refusing to solve X.

---

## The dataset

Pairs are JSON objects in `data/*.json`. Only the fields fed to the solver are shown to
models; the rest is curation metadata.

```json
{
  "id": "hard_007",
  "x": "... question X ...",
  "y": "... question Y ...",
  "answer_x": "8/27",
  "answer_y": "10/27",
  "relation": "Method-level description of why X's natural trace leaks Y and what the alternate non-leaking method is. (Fed verbatim to the cheating_avoid scaffold, so it must NOT contain answers or worked arithmetic.)",
  "y_proximity_signals": ["intermediate quantities in X's trace that give Y away"],
  "flavor": "hard_probability",
  "gold_solution": "reference worked solution (not shown to solvers)",
  "rubric_x": "grading note",
  "rubric_y": "grading note"
}
```

`flavor`, `gold_solution`, `rubric_x`, and `rubric_y` are optional metadata. The `rubric_*`
fields and the `RubricGrader` stub exist so the dataset can later move from
single-checkable-answer grading to open-ended (rubric / LLM-judge) grading without reshaping
the pairs.

> **Important:** `relation` is injected directly into the `cheating_avoid` prompt. Keep it
> qualitative — no computed values and no answers — or it spoils X and risks leaking Y. The
> numbers live in `gold_solution` / `y_proximity_signals`, which are never shown to solvers.

### Pair sets

- `data/pairs.json` — the original proof-of-concept set.
- `data/pairs_generated_v1.json` — auto-generated via the generation pipeline.
- `data/pairs_hard_v4.json` — hand-authored "harder, less toy" set (this README's results),
  12 pairs across four flavors: `competition_math`, `hard_probability`,
  `applied_derivation`, `graduate_mixed`.

### Curating pairs

Two paths feed the dataset, both ending in the same vetting bar:

- **Hand-authoring** — write a pair, then `vet` it (verify the gold answers re-solve, then
  measure leakage against the desiderata thresholds). This is the primary path for hard pairs.
- **Generation** — auto-propose candidates, verify, measure, and filter, then hand-review the
  survivors.

```bash
python main.py vet --pairs-file data/pairs_hard_v4.json
# --trace-model solver  to produce the X trace with the cheap solver instead of the strong model
```

`vet` writes a `<pairs>.vet.json` report with per-pair and per-flavor leakage, scratch
accuracy, and pass/fail against the curation thresholds.

---

## Scaffolds under test

Once the dataset exists, we evaluate different **scaffolds** against it. A scaffold is a
strategy for producing X's trace; they range from a plain solve to ones that actively try to
avoid helping Y. They are the *subjects* of the eval — the dataset is what judges them.

| Scaffold | What it does |
|---|---|
| `control` | Plain solve of X; the entire response becomes the trace. The default leakage baseline. |
| `control_answer_only` | Only X's final answer is shown as the "trace." Baseline for *answer-only* leakage. |
| `baseline_avoid` | Solver is told to solve X while not helping Y — but given no guidance on how. |
| `cheating_avoid` | Like `baseline_avoid`, but additionally handed the pair's `relation` hint (which names the leak and an alternate non-leaking method). An *informed upper bound* on avoidance. |
| `threaded` | Decomposes X into steps; a monitor scores each step for leakage and cuts/reworks risky ones. |
| `two_agent` | Solves X one step at a time, each step instructed to avoid Y-relevant content. |

---

## Models

Three roles, configured via environment variables (see [Setup](#setup)):

- **Solver / target** (`SOLVER_*`, default `openai/gpt-4o-mini`) — the model whose Y accuracy
  defines "hard," and which answers Y. Cheap and weak on purpose.
- **Grader** (`GRADER_*`, default `gpt-4o-mini`) — checks answers (exact rules first, then an
  LLM verdict fallback).
- **Generation / strong** (`GENERATION_*`, default `anthropic/claude-opus-4-8`) — generates
  and verifies candidate pairs, and (in `vet`) produces the *correct* X trace to leak from.

The `run` command lets X and Y use **different** models: `--x-model` / `--x-provider` set the
model that solves X, while `--model` answers Y. If `--x-model` is unset, X uses `--model`
(one model for both). A *smart X, weak Y* split maximizes measurable leakage.

---

## Repository layout

```
main.py              CLI entry point (run / analyze / y-sweep / visualize / vet / generate)
config.py            Defaults + env-var configuration
models.py            Pydantic data models (QuestionPair, LeakageResult, RunSummary, ...)
data/                Question pair sets and their vet/metrics reports  <- the core artifact
generation/
  pipeline.py        Auto-generate -> verify -> measure -> filter pairs
  generate.py        Candidate generation prompts/calls
  verify.py          Re-solve candidates to confirm gold answers
  measure.py         Measure scratch / with-trace / answer-only Y accuracy
  scoring.py         Curation thresholds + structural-triviality guard
  vet.py             Verify + measure a hand-authored pairs file
eval/
  grader.py          Answer grading (Grader interface: ExactAnswerGrader, RubricGrader stub)
  y_leakage.py       Y sweeps, accuracy aggregation, leakage metrics
  runner.py          Full eval orchestration, summary building, printing
scaffolds/           One module per scaffold + shared base helpers (the test subjects)
visualizer/          Local web UI for browsing runs, traces, and Y curves
runs/                Per-run outputs (manifest, traces, Y trials, summary)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` with API keys and (optionally) model overrides:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Optional overrides (defaults shown)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
GENERATION_PROVIDER=anthropic
GENERATION_MODEL=claude-opus-4-8
GRADER_PROVIDER=openai
GRADER_MODEL=gpt-4o-mini
SOLVER_PROVIDER=openai
SOLVER_MODEL=gpt-4o-mini
```

---

## Usage

### Vet / curate the dataset

```bash
python main.py vet --pairs-file data/pairs_hard_v4.json
python main.py generate --target 160      # auto-generate a new candidate pair set
```

### Run a full evaluation

Solve X with each scaffold, then measure Y leakage:

```bash
python main.py run --pairs-file data/pairs_hard_v4.json --fast
```

Smart X, weak Y (the recommended split for measuring leakage):

```bash
python main.py run --pairs-file data/pairs_hard_v4.json \
  --provider openai --model gpt-4o-mini \
  --x-provider anthropic --x-model claude-opus-4-8
```

Useful flags: `--scaffolds a,b,c`, `--pairs id1,id2`, `--trials`, `--y-trials`,
`--budgets 32,64,...`, `--skip-y`, `--fast`.

### Inspect results

```bash
python main.py analyze --run-id <id>      # re-grade + rebuild a run's summary
python main.py y-sweep --trace <path>     # Y sweep for a single trace file
python main.py visualize                  # http://127.0.0.1:8765
```

---

## Results

### Vetting the hard set (`pairs_hard_v4.json`)

All 12 pairs verify (gold answers re-solve). Leakage below is measured with a **strong,
correct X trace** (Opus) and a **weak Y solver** (`gpt-4o-mini`) over the fast budget ladder.
A pair "passes" the curation desiderata when Y is hard from scratch, trace leakage is high,
and answer-only leakage is low.

| Flavor | Raw trace leakage | Normalized | Y scratch acc | Passes |
|---|---|---|---|---|
| `hard_probability` | +0.47 | ~57% | 0.18 | 2 / 4 |
| `applied_derivation` | +0.48 | ~50% | 0.04 | 1 / 2 |
| `graduate_mixed` | +0.29 | ~54% | 0.46 | 0 / 3 |
| `competition_math` | +0.14 | ~17% | 0.18 | 1 / 4 |

Findings (these are statements about the **dataset**, i.e. which pair styles satisfy the
desiderata best):

- **`hard_probability` is the strongest flavor** — "same setup, different target" pairs
  (Markov stationary distribution, gambler's ruin, Bayes denominator) leak reliably because
  the natural full solve of X computes a reusable artifact that X's bare answer hides.
- **The strongest single pairs** are `hard_007` (Markov stationary, +0.92 in a later run),
  `hard_009` (circuit power), `hard_005` (gambler's ruin), and `hard_004` (divisor count).
- **`graduate_mixed` leaks well but its Y questions are too easy from scratch** (Y scratch
  0.46), so the pairs fail desideratum #1 even though the trace clearly helps.
- **`competition_math` is mixed**: Y is satisfyingly hard, but several pairs' traces don't
  transfer the specific artifact Y needs (desideratum #3), so leakage is near zero.
- Leakage is bounded by `1 - Y_scratch`. If the Y solver is too strong for a pair, that pair
  cannot show leakage regardless of trace quality.

### The ceiling effect

A `run` that (inadvertently) answered **Y with Opus** instead of the weak solver produced raw
control leakage of only **9%** — not because the pairs are weak, but because Opus already
solved Y ~88% of the time from scratch, leaving almost no headroom. Normalized, that same 9%
is **76% of the available headroom**. This is exactly why we report normalized leakage and
why the eval wants a *weak* Y solver paired with a *smart* X model.

---

## Notes and caveats

- Small `--trials` / `--y-trials` (e.g. the `--fast` defaults of 1 / 2) make per-pair numbers
  noisy; raise them before drawing conclusions about a pair.
- Gold answers are kept to single checkable values (integers, exact fractions, short decimals)
  so the rule-based verifier can confirm them. Decimal answers must match closely, so prefer
  exact rationals where possible.
- Generation/curation thresholds (`generation/scoring.py`) are calibrated to **raw** leakage,
  not normalized.
