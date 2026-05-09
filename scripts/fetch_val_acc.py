#!/usr/bin/env python3
"""
fetch_val_acc.py

Reads swanlab backup.swanlab files for every experiment listed in the
runbook's "Live SwanLab Log Map" and prints benchmark accuracy tables
(AIME24 / AIME25 / AMC23, best@2 and maj@2).

Usage examples:
    # Current status of all runs (training steps reached + val steps available)
    python3 fetch_val_acc.py --status

    # Summary: latest val step per experiment (annotated with which step)
    python3 fetch_val_acc.py

    # Compare all experiments at a fixed step (e.g. step 10, the only common step)
    python3 fetch_val_acc.py --step 10

    # Full per-step tables for every experiment
    python3 fetch_val_acc.py --all-steps

    # Write a full markdown report to distll-best/
    python3 fetch_val_acc.py --report

    # Write report to a custom path
    python3 fetch_val_acc.py --report --out /path/to/report.md

    # Include 'mean@2' in addition to best/maj
    python3 fetch_val_acc.py --metrics best maj mean
"""

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

# ── Experiment registry (matches runbook "Live SwanLab Log Map") ──────────────
# Override via env var: SWAN_BASE=/path/to/swanlab_log python fetch_val_acc.py ...
SWAN_BASE = os.environ.get(
    "SWAN_BASE",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoint",
        "swanlab_log",
    ),
)

EXPERIMENTS = [
    {
        "name": "E1 (OPD-like, rollout=off)",
        "swan_dir": "2026-04-24_17-37-02_1839927",
    },
    {
        "name": "E7b (rollout=1, tier1)",
        "swan_dir": "2026-04-24_18-13-01_1904764",
    },
    {
        "name": "rollout=2 (teacher_topk, tier1)",
        "swan_dir": "2026-04-24_17-53-40_169054",  # val_log ts matches; runbook pointed to wrong dir 17-54-27_17313
    },
    {
        "name": "rollout=4 (teacher_topk, tier1)",
        "swan_dir": "2026-04-24_17-55-24_2569327",
    },
    {
        "name": "tier2 + rollout=2",
        "swan_dir": "2026-04-24_17-55-34_2252680",
    },
    {
        "name": "tier2 + rollout=4",
        "swan_dir": "2026-04-24_17-55-55_302057",
    },
    {
        "name": "reward ablation (RM=7B)",
        "swan_dir": "2026-04-24_21-30-54_2390106",
    },
    {
        "name": "apply disturb (coef=1)",
        "swan_dir": "2026-04-24_19-42-00_802871",  # val_log ts matches; runbook log map pointed to wrong dir 17-40-25_494549
    },
    {
        "name": "RM-changeopd (n=2, rollout=off)",
        "swan_dir": "2026-04-24_19-04-21_200692",
    },
]

BENCHMARKS = ["AIME24", "AIME25", "AMC23"]

# ── swanlab parsing ───────────────────────────────────────────────────────────

def find_latest_run(swan_dir_path):
    if not os.path.isdir(swan_dir_path):
        return None
    runs = [
        d for d in os.listdir(swan_dir_path)
        if d.startswith("run-") and os.path.isdir(os.path.join(swan_dir_path, d))
    ]
    if not runs:
        return None
    runs.sort(key=lambda d: os.path.getmtime(os.path.join(swan_dir_path, d)), reverse=True)
    return os.path.join(swan_dir_path, runs[0], "backup.swanlab")


def parse_backup(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        raw = f.read()
    metrics = defaultdict(list)
    for line in raw.split(b"\n"):
        idx = line.find(b"{")
        if idx == -1:
            continue
        try:
            obj = json.loads(line[idx:])
        except Exception:
            continue
        if obj.get("model_type") != "Scalar":
            continue
        data = obj.get("data", {})
        key = data.get("key", "")
        step = data.get("step")
        value = data.get("metric", {}).get("data")
        if key and step is not None and value is not None:
            metrics[key].append((step, value))
    return {k: sorted(v, key=lambda x: x[0]) for k, v in metrics.items()}


def load_all():
    result = []
    for exp in EXPERIMENTS:
        swan_path = os.path.join(SWAN_BASE, exp["swan_dir"])
        backup = find_latest_run(swan_path)
        metrics = parse_backup(backup)
        result.append({**exp, "metrics": metrics})
    return result


# ── metric key helpers ────────────────────────────────────────────────────────

def build_keys(families):
    keys = []
    for bench in BENCHMARKS:
        for fam in families:
            if fam == "best":
                mk = f"val-aux/{bench}/acc/best@2/mean"
                label = f"{bench} best@2"
            elif fam == "maj":
                mk = f"val-aux/{bench}/acc/maj@2/mean"
                label = f"{bench} maj@2"
            elif fam == "mean":
                mk = f"val-aux/{bench}/acc/mean@2"
                label = f"{bench} mean@2"
            elif fam == "best4":
                mk = f"val-core/{bench}/acc/best@4/mean"
                label = f"{bench} best@4"
            elif fam == "maj4":
                mk = f"val-core/{bench}/acc/maj@4/mean"
                label = f"{bench} maj@4"
            elif fam == "mean4":
                mk = f"val-core/{bench}/acc/mean@4"
                label = f"{bench} mean@4"
            else:
                continue
            keys.append((label, mk))
    return keys


def val_steps_for(metrics, metric_keys):
    return sorted(set(
        s for _, mk in metric_keys for s, _ in metrics.get(mk, [])
    ))


def max_train_step(metrics):
    train_keys = [k for k in metrics if any(x in k for x in ("actor", "train", "rollout"))]
    steps = [pts[-1][0] for k in train_keys for pts in [metrics[k]] if pts]
    return max(steps, default=None)


# ── formatting ────────────────────────────────────────────────────────────────

def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def lookup(metrics, mk, step):
    d = {s: v for s, v in metrics.get(mk, [])}
    v = d.get(step)
    # val-core/*/acc/mean@4 has a logging gap at step 10 for some runs;
    # fall back to val-aux/*/score/mean@4 which carries identical values
    if v is None and mk.startswith("val-core/") and mk.endswith("/acc/mean@4"):
        bench = mk.split("/")[1]
        fallback = f"val-aux/{bench}/score/mean@4"
        d2 = {s: v2 for s, v2 in metrics.get(fallback, [])}
        v = d2.get(step)
    return v


def render_table(rows, headers, md=False, out=None):
    p = out.write if out else sys.stdout.write
    nl = "\n"
    if md:
        p("| " + " | ".join(headers) + " |" + nl)
        p("| " + " | ".join(["---"] * len(headers)) + " |" + nl)
        for row in rows:
            p("| " + " | ".join(row) + " |" + nl)
    else:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
        p(line(headers) + nl)
        p("  ".join("-" * w for w in widths) + nl)
        for row in rows:
            p(line(row) + nl)


# ── modes ─────────────────────────────────────────────────────────────────────

def mode_status(data, out=None):
    p = (lambda s: out.write(s)) if out else print
    p(f"\n{'Experiment':<42} {'Max train step':>14}  {'Val steps with acc data'}")
    p("-" * 90)
    for exp in data:
        m = exp["metrics"]
        mt = max_train_step(m)
        vsteps = val_steps_for(m, build_keys(["best"]))
        vsteps_str = str(vsteps) if vsteps else "(none yet)"
        p(f"  {exp['name']:<40} {str(mt or '—'):>14}  {vsteps_str}")


def mode_summary(data, metric_keys, md, out=None):
    p = (lambda s: out.write(s + "\n")) if out else print
    p(f"\nLatest val step per experiment  —  val-aux @2\n"
      f"(runs are at different training stages; see --status or --step N for fair comparison)\n")
    headers = ["Experiment"] + [label for label, _ in metric_keys] + ["val step"]
    rows = []
    for exp in data:
        vsteps = val_steps_for(exp["metrics"], metric_keys)
        latest = vsteps[-1] if vsteps else None
        row = [exp["name"]]
        for label, mk in metric_keys:
            v = lookup(exp["metrics"], mk, latest) if latest else None
            row.append(fmt(v))
        row.append(str(latest) if latest else "—")
        rows.append(row)
    render_table(rows, headers, md, out)


def mode_at_step(data, step, metric_keys, md, out=None):
    p = (lambda s: out.write(s + "\n")) if out else print
    p(f"\nAll experiments at step {step}  —  val-aux @2\n")
    headers = ["Experiment"] + [label for label, _ in metric_keys]
    rows = []
    for exp in data:
        row = [exp["name"]]
        for label, mk in metric_keys:
            row.append(fmt(lookup(exp["metrics"], mk, step)))
        rows.append(row)
    render_table(rows, headers, md, out)


def mode_all_steps(data, metric_keys, md, out=None):
    p = (lambda s: out.write(s + "\n")) if out else print
    for exp in data:
        vsteps = val_steps_for(exp["metrics"], metric_keys)
        mt = max_train_step(exp["metrics"])
        sep = "=" * 62
        p(f"\n{sep}")
        p(f"  {exp['name']}")
        p(f"  dir: {exp['swan_dir']}")
        p(f"  max training step: {mt or '—'}   |   val steps: {vsteps or '(none)'}")
        p(sep)
        if not vsteps:
            p("  (no val-aux accuracy metrics logged yet)")
            continue
        headers = ["step"] + [label for label, _ in metric_keys]
        rows = []
        for s in vsteps:
            row = [str(s)] + [fmt(lookup(exp["metrics"], mk, s)) for _, mk in metric_keys]
            rows.append(row)
        render_table(rows, headers, md, out)


def _macro_best(metrics, metric_keys, step):
    """Average of best@ values across all benchmarks at a given step."""
    vals = []
    for label, mk in metric_keys:
        if not any(f"best@{n}" in label for n in ("2", "4")):
            continue
        v = lookup(metrics, mk, step)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def mode_report(data, metric_keys, out_path):
    """Write a full markdown report to out_path."""
    buf = io.StringIO()
    w = buf.write
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    w(f"# Val Accuracy Report\n\n")
    w(f"Generated: {now}  \n")
    w(f"Source: `OPD_verify/checkpoint/swanlab_log/*/backup.swanlab`  \n")
    w(f"Metrics: val-aux @2 student samples (AIME24, AIME25, AMC23)\n\n")
    w(f"> **val-aux @2**: AIME24/AIME25/AMC23 with n=2 student samples (best@2 / maj@2).\n")
    w(f"> **val-core @4**: same benchmarks with n=4 student samples (mean@4 / best@4 / maj@4).\n")
    w(f"> Step-10 is the only common point across most runs — use that for fair comparisons.\n\n")
    w("---\n\n")

    # ── 1. Status ──────────────────────────────────────────────────────────────
    w("## 1. Training Progress\n\n")
    w("| Experiment | Max train step | Val steps with acc data |\n")
    w("| --- | --- | --- |\n")
    for exp in data:
        m = exp["metrics"]
        mt = max_train_step(m)
        vsteps = val_steps_for(m, build_keys(["best"]))
        vsteps_str = str(vsteps) if vsteps else "*(none yet)*"
        w(f"| {exp['name']} | {mt or '—'} | {vsteps_str} |\n")
    w("\n")

    # ── 2. Fair comparison at step 10 ─────────────────────────────────────────
    w("## 2. Comparison at Step 10 (fairest common checkpoint)\n\n")
    headers = ["Experiment"] + [label for label, _ in metric_keys] + ["Macro best@2"]
    w("| " + " | ".join(headers) + " |\n")
    w("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for exp in data:
        row = [exp["name"]]
        for label, mk in metric_keys:
            row.append(fmt(lookup(exp["metrics"], mk, 10)))
        macro = _macro_best(exp["metrics"], metric_keys, 10)
        row.append(fmt(macro))
        w("| " + " | ".join(row) + " |\n")
    w("\n")

    # ── 2b. Same-step comparison at 10, 20, 30 ───────────────────────────────
    w("## 2b. Same-Step Comparison\n\n")
    w("> Only experiments that have logged val metrics at each step are shown; others get `—`.\n\n")
    best_keys = [(label, mk) for label, mk in metric_keys if "best@2" in label]
    for step in (10, 20, 30):
        w(f"### Step {step}\n\n")
        has_any = any(
            lookup(exp["metrics"], mk, step) is not None
            for exp in data for _, mk in metric_keys
        )
        if not has_any:
            w("*(no experiments have val data at this step yet)*\n\n")
            continue
        headers = ["Experiment"] + [label for label, _ in metric_keys] + ["Macro best@2"]
        w("| " + " | ".join(headers) + " |\n")
        w("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for exp in data:
            row = [exp["name"]]
            for label, mk in metric_keys:
                row.append(fmt(lookup(exp["metrics"], mk, step)))
            macro = _macro_best(exp["metrics"], best_keys, step)
            row.append(fmt(macro))
            w("| " + " | ".join(row) + " |\n")
        w("\n")

    # ── 2c. val-core @4 same-step comparison ─────────────────────────────────
    core_families = ["mean4", "best4", "maj4"]
    core_keys = build_keys(core_families)
    core_best_keys = [(l, mk) for l, mk in core_keys if "best@4" in l]
    w("## 2c. Same-Step Comparison — val-core @4\n\n")
    w("> `mean@4` = average pass rate over 4 student responses per problem.  \n")
    w("> `best@4` = probability at least one of 4 responses is correct.  \n")
    w("> `maj@4` = majority-vote accuracy over 4 responses.\n\n")
    for step in (10, 20, 30):
        w(f"### Step {step}\n\n")
        has_any = any(
            lookup(exp["metrics"], mk, step) is not None
            for exp in data for _, mk in core_keys
        )
        if not has_any:
            w("*(no experiments have val-core data at this step yet)*\n\n")
            continue
        headers = ["Experiment"] + [label for label, _ in core_keys] + ["Macro best@4"]
        w("| " + " | ".join(headers) + " |\n")
        w("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for exp in data:
            row = [exp["name"]]
            for label, mk in core_keys:
                row.append(fmt(lookup(exp["metrics"], mk, step)))
            macro = _macro_best(exp["metrics"], core_best_keys, step)
            row.append(fmt(macro))
            w("| " + " | ".join(row) + " |\n")
        w("\n")

    # ── 2d. Progression grid: experiments × steps for mean@4 and maj@4 ────────
    w("## 2d. Progression Grid — mean@4 and maj@4\n\n")
    w("> Rows = experiments, columns = training steps. Each cell is the metric at that checkpoint.\n\n")
    grid_fams = [("mean@4", "mean4"), ("maj@4", "maj4")]
    # collect all steps that have any val-core data across all experiments
    all_core_steps = sorted(set(
        s for exp in data
        for _, mk in build_keys(["mean4"])
        for s, _ in exp["metrics"].get(mk, [])
    ))
    if not all_core_steps:
        w("*(no val-core data yet)*\n\n")
    else:
        step_headers = ["Experiment"] + [f"step {s}" for s in all_core_steps]
        for bench in BENCHMARKS:
            w(f"### {bench}\n\n")
            for metric_label, fam in grid_fams:
                mk = build_keys([fam])  # list of (label, key) for all benches
                # find the key for this specific bench
                bench_mk = next((k for l, k in mk if bench in l), None)
                if bench_mk is None:
                    continue
                w(f"**{metric_label}**\n\n")
                w("| " + " | ".join(step_headers) + " |\n")
                w("| " + " | ".join(["---"] * len(step_headers)) + " |\n")
                for exp in data:
                    row = [exp["name"]]
                    for s in all_core_steps:
                        row.append(fmt(lookup(exp["metrics"], bench_mk, s)))
                    w("| " + " | ".join(row) + " |\n")
                w("\n")
        # macro (average across all 3 benchmarks)
        w("### Macro (avg across AIME24 / AIME25 / AMC23)\n\n")
        for metric_label, fam in grid_fams:
            fam_keys = build_keys([fam])
            w(f"**{metric_label}**\n\n")
            w("| " + " | ".join(step_headers) + " |\n")
            w("| " + " | ".join(["---"] * len(step_headers)) + " |\n")
            for exp in data:
                row = [exp["name"]]
                for s in all_core_steps:
                    vals = [v for _, mk in fam_keys
                            if (v := lookup(exp["metrics"], mk, s)) is not None]
                    row.append(fmt(sum(vals) / len(vals) if vals else None))
                w("| " + " | ".join(row) + " |\n")
            w("\n")

    # ── 3. Latest snapshot ────────────────────────────────────────────────────
    w("## 3. Latest Available Snapshot (different steps — not directly comparable)\n\n")
    headers = ["Experiment"] + [label for label, _ in metric_keys] + ["Macro best@2", "val step"]
    w("| " + " | ".join(headers) + " |\n")
    w("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for exp in data:
        vsteps = val_steps_for(exp["metrics"], metric_keys)
        latest = vsteps[-1] if vsteps else None
        row = [exp["name"]]
        for label, mk in metric_keys:
            v = lookup(exp["metrics"], mk, latest) if latest else None
            row.append(fmt(v))
        macro = _macro_best(exp["metrics"], metric_keys, latest) if latest else None
        row.append(fmt(macro))
        row.append(str(latest) if latest else "—")
        w("| " + " | ".join(row) + " |\n")
    w("\n")

    # ── 4. Per-experiment full history ────────────────────────────────────────
    core_families = ["mean4", "best4", "maj4"]
    core_keys = build_keys(core_families)
    core_best_keys = [(l, mk) for l, mk in core_keys if "best@4" in l]
    w("## 4. Per-Experiment Full History\n\n")
    for exp in data:
        all_steps = sorted(set(
            val_steps_for(exp["metrics"], metric_keys) +
            val_steps_for(exp["metrics"], core_keys)
        ))
        mt = max_train_step(exp["metrics"])
        w(f"### {exp['name']}\n\n")
        w(f"- SwanLab dir: `{exp['swan_dir']}`\n")
        w(f"- Max training step reached: **{mt or '—'}**\n")
        w(f"- Val checkpoints: {all_steps if all_steps else '*(none yet)*'}\n\n")
        if not all_steps:
            w("*(No accuracy metrics logged yet.)*\n\n")
            continue
        w("#### val-aux @2\n\n")
        headers = ["step"] + [label for label, _ in metric_keys] + ["Macro best@2"]
        w("| " + " | ".join(headers) + " |\n")
        w("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for s in all_steps:
            row = [str(s)]
            for label, mk in metric_keys:
                row.append(fmt(lookup(exp["metrics"], mk, s)))
            macro = _macro_best(exp["metrics"], metric_keys, s)
            row.append(fmt(macro))
            w("| " + " | ".join(row) + " |\n")
        w("\n")
        w("#### val-core @4\n\n")
        headers = ["step"] + [label for label, _ in core_keys] + ["Macro best@4"]
        w("| " + " | ".join(headers) + " |\n")
        w("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for s in all_steps:
            row = [str(s)]
            for label, mk in core_keys:
                row.append(fmt(lookup(exp["metrics"], mk, s)))
            macro = _macro_best(exp["metrics"], core_best_keys, s)
            row.append(fmt(macro))
            w("| " + " | ".join(row) + " |\n")
        w("\n")

    content = buf.getvalue()
    with open(out_path, "w") as f:
        f.write(content)
    print(f"Report written to: {out_path}")
    # also print a preview of the status table
    print("\nPreview — Training Progress:")
    for exp in data:
        m = exp["metrics"]
        mt = max_train_step(m)
        vsteps = val_steps_for(m, build_keys(["best"]))
        print(f"  {exp['name']:<42}  step {mt or '—':>4}  val={vsteps or '[]'}")


# ── main ──────────────────────────────────────────────────────────────────────

DEFAULT_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "val_acc_report.md",
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--status", action="store_true",
                        help="Show training progress and val steps available per run")
    parser.add_argument("--step", type=int, default=None,
                        help="Compare all experiments at this specific training step")
    parser.add_argument("--all-steps", action="store_true",
                        help="Show full per-step table for every experiment")
    parser.add_argument("--report", action="store_true",
                        help="Write a full markdown report (default path: distll-best/val_acc_report.md)")
    parser.add_argument("--out", default=None,
                        help="Output path for --report (overrides default)")
    parser.add_argument("--metrics", nargs="+", default=["best", "maj"],
                        choices=["best", "maj", "mean", "best4", "maj4", "mean4"],
                        help="Metric families to include (default: best maj); "
                             "best/maj/mean use val-aux @2; best4/maj4/mean4 use val-core @4")
    parser.add_argument("--md", action="store_true",
                        help="Output markdown tables (for stdout modes)")
    args = parser.parse_args()

    data = load_all()
    metric_keys = build_keys(args.metrics)

    if args.report:
        out_path = args.out or DEFAULT_REPORT_PATH
        mode_report(data, metric_keys, out_path)
    elif args.status:
        mode_status(data)
    elif args.all_steps:
        mode_all_steps(data, metric_keys, args.md)
    elif args.step is not None:
        mode_at_step(data, args.step, metric_keys, args.md)
    else:
        mode_summary(data, metric_keys, args.md)


if __name__ == "__main__":
    main()
