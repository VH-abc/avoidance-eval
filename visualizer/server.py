from __future__ import annotations

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from config import RUNS_DIR, TOKEN_BUDGETS, resolve_pairs_path
from eval.y_leakage import accuracies_from_results, leakage_from_accuracies
from models import YTrialResult
from scaffolds.base import load_pair_for_run, load_pairs_for_run

STATIC_DIR = Path(__file__).parent / "static"
TRIAL_RE = re.compile(r"^(\d+)\.json$")


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs: list[dict] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        entry = {"run_id": run_dir.name}
        if manifest_path.exists():
            entry["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if summary_path.exists():
            entry["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append(entry)
    return runs


def build_run_tree(run_id: str) -> dict:
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_id)

    tree: dict[str, dict[str, list[int]]] = {}
    for pair_dir in sorted(run_dir.iterdir()):
        if not pair_dir.is_dir() or pair_dir.name.startswith("_"):
            continue
        pair_id = pair_dir.name
        tree[pair_id] = {}
        for scaffold_dir in sorted(pair_dir.iterdir()):
            if not scaffold_dir.is_dir():
                continue
            trials: list[int] = []
            for path in scaffold_dir.glob("*.json"):
                match = TRIAL_RE.match(path.name)
                if match:
                    trials.append(int(match.group(1)))
            if trials:
                tree[pair_id][scaffold_dir.name] = sorted(trials)
    return tree


def find_scratch_path(pair_dir: Path) -> Path | None:
    matches = sorted(pair_dir.glob("_scratch_y*.json"))
    if not matches:
        legacy = pair_dir / "_scratch_y.json"
        if legacy.exists():
            return legacy
        return None
    return matches[-1]


def build_run_entries(run_id: str) -> list[dict]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_id)

    token_budgets = TOKEN_BUDGETS
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        token_budgets = manifest.get("token_budgets", TOKEN_BUDGETS)

    entries: list[dict] = []
    for pair_id, scaffolds in build_run_tree(run_id).items():
        for scaffold, trials in scaffolds.items():
            for trial in trials:
                result_path = run_dir / pair_id / scaffold / f"{trial}.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                y_path = run_dir / pair_id / scaffold / f"{trial}_y_trials.json"

                entry: dict = {
                    "pair_id": pair_id,
                    "scaffold": scaffold,
                    "trial": trial,
                    "answer_x": result.get("answer_x"),
                    "trace_length": len(result.get("trace", [])),
                    "y_accuracy_scratch": None,
                    "y_accuracy_with_trace": None,
                    "leakage": None,
                }

                if y_path.exists():
                    y_data = json.loads(y_path.read_text(encoding="utf-8"))
                    results = [YTrialResult.model_validate(item) for item in y_data]
                    scratch = [item for item in results if item.condition == "scratch"]
                    trace_results = [item for item in results if item.condition == "with_trace"]

                    y_accuracy_scratch, y_accuracy_with_trace, leakage = leakage_from_accuracies(
                        accuracies_from_results(scratch),
                        accuracies_from_results(trace_results),
                        token_budgets,
                    )
                    entry["y_accuracy_scratch"] = y_accuracy_scratch
                    entry["y_accuracy_with_trace"] = y_accuracy_with_trace
                    entry["leakage"] = leakage

                entries.append(entry)
    return entries


def merge_y_trials(y_trials: list[dict] | None, scratch_y: list[dict] | None) -> list[dict]:
    if not y_trials and not scratch_y:
        return []
    if not y_trials:
        return list(scratch_y or [])
    if not scratch_y:
        return list(y_trials)
    has_scratch = any(item.get("condition") == "scratch" for item in y_trials)
    if has_scratch:
        return list(y_trials)
    return list(scratch_y) + list(y_trials)


def pairs_file_for_run(run_id: str) -> str | None:
    manifest_path = RUNS_DIR / run_id / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("pairs_file")


def load_result(run_id: str, pair_id: str, scaffold: str, trial: int) -> dict:
    run_dir = RUNS_DIR / run_id
    result_path = run_dir / pair_id / scaffold / f"{trial}.json"
    if not result_path.exists():
        raise FileNotFoundError(str(result_path))

    result = json.loads(result_path.read_text(encoding="utf-8"))
    y_path = run_dir / pair_id / scaffold / f"{trial}_y_trials.json"
    scratch_path = find_scratch_path(run_dir / pair_id)

    payload = {"result": result, "y_trials": None, "scratch_y": None, "pair": None}
    y_trials = None
    scratch_y = None
    if y_path.exists():
        y_trials = json.loads(y_path.read_text(encoding="utf-8"))
    if scratch_path and scratch_path.exists():
        scratch_y = json.loads(scratch_path.read_text(encoding="utf-8"))
    merged = merge_y_trials(y_trials, scratch_y)
    if merged:
        payload["y_trials"] = merged
    if scratch_y:
        payload["scratch_y"] = scratch_y

    pair = load_pair_for_run(run_id, pair_id)
    if pair is not None:
        payload["pair"] = pair.model_dump()
    return payload


def compute_y_accuracy_curve(trials: list[dict]) -> list[dict]:
    by_budget: dict[int, dict[str, list[bool]]] = {}
    for item in trials:
        budget = item["budget"]
        condition = item["condition"]
        by_budget.setdefault(budget, {}).setdefault(condition, []).append(item["correct"])

    points: list[dict] = []
    for budget in sorted(by_budget):
        point = {"budget": budget}
        for condition, values in by_budget[budget].items():
            point[condition] = sum(values) / len(values) if values else 0.0
        points.append(point)
    return points


class VisualizerHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/runs":
            self._send_json(list_runs())
            return

        if path == "/api/pairs":
            run_id = parse_qs(parsed.query).get("run_id", [None])[0]
            if run_id:
                pairs = [pair.model_dump() for pair in load_pairs_for_run(run_id)]
            else:
                pairs_path = resolve_pairs_path(None)
                pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
            self._send_json(pairs)
            return

        if path.startswith("/api/runs/"):
            parts = path.split("/")
            if len(parts) >= 4:
                run_id = parts[3]
                if len(parts) == 5 and parts[4] == "tree":
                    self._send_json(build_run_tree(run_id))
                    return
                if len(parts) == 5 and parts[4] == "summary":
                    summary_path = RUNS_DIR / run_id / "summary.json"
                    if summary_path.exists():
                        self._send_json(json.loads(summary_path.read_text(encoding="utf-8")))
                    else:
                        self.send_error(404)
                    return
                if len(parts) == 5 and parts[4] == "entries":
                    self._send_json(build_run_entries(run_id))
                    return
                if len(parts) == 8 and parts[4] == "result":
                    pair_id, scaffold, trial_str = parts[5], parts[6], parts[7]
                    payload = load_result(run_id, pair_id, scaffold, int(trial_str))
                    if payload["y_trials"]:
                        payload["y_curve"] = compute_y_accuracy_curve(payload["y_trials"])
                    self._send_json(payload)
                    return

        static_path = STATIC_DIR / (path.lstrip("/") or "index.html")
        if static_path.is_file():
            content_type = mimetypes.guess_type(str(static_path))[0] or "application/octet-stream"
            self._send_bytes(static_path.read_bytes(), content_type)
            return

        self.send_error(404)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), VisualizerHandler)
    print(f"Thread Cutter visualizer: http://{host}:{port}", flush=True)
    server.serve_forever()
