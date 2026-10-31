#!/usr/bin/env python
"""
Run the whole project with one command and see every step clearly.

    python run_project.py                 # resume: run only what is missing, then reports + tests
    python run_project.py --dry-run       # show the plan, run nothing
    python run_project.py --list          # list the stages
    python run_project.py --only reports  # run selected stages (comma-separated keys)
    python run_project.py --force --yes   # redo EVERY stage from scratch (long; see HOW_IT_WORKS.md)

Default behaviour is safe: a stage whose output already exists is skipped, so the held-out
test set is never re-evaluated by accident (the project evaluates each model on it once).
The report stages (tables, plots, diagrams) and the test suite run every time.

Each stage's full output is saved to logs/run_<timestamp>/<stage>.log. Results are summarised
in the terminal and in results/run_report.md. How the data flows: HOW_IT_WORKS.md.
"""

import argparse
import csv
import importlib.metadata as md
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.config import DEFAULT_CONFIG_PATH, Config, load_config  # noqa: E402  (needs ROOT on sys.path)

RAW_FILENAMES = [
    "Benign-Monday-no-metadata.parquet", "Bruteforce-Tuesday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet", "WebAttacks-Thursday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet", "Botnet-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet", "DDoS-Friday-no-metadata.parquet",
]
PACKAGES = ["torch", "flwr", "scikit-learn", "optuna", "pandas", "numpy", "pyarrow", "matplotlib", "joblib", "PyYAML", "pytest"]
# Lines worth showing from long training stages when --verbose is off (everything is still logged).
KEY_LINES = re.compile(
    r"^\[round|^\[trial|^epoch|^Study|^Best trial|^Retraining|Training time|HELD-OUT|Validation \(|Done in|"
    r"^Saved|Attack recall|False (neg|pos)|Accuracy|ROC-AUC|Confusion|act (Benign|Attack)|pred Benign|"
    r"^Model:|^Rounds:|^Device|^Train rows|Loaded|Created|Error|Traceback|refus", re.I)


def rel(p: Path) -> Path:
    """Path relative to the project root when inside it, else unchanged."""
    try:
        return p.resolve().relative_to(ROOT)
    except ValueError:
        return p


# ----------------------------------------------------------------------------- terminal styling
class Style:
    def __init__(self, enabled: bool) -> None:
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    bold = lambda self, s: self._w("1", s)
    dim = lambda self, s: self._w("2", s)
    red = lambda self, s: self._w("31", s)
    green = lambda self, s: self._w("32", s)
    yellow = lambda self, s: self._w("33", s)
    blue = lambda self, s: self._w("34;1", s)
    cyan = lambda self, s: self._w("36", s)


S = Style(False)
WIDTH = 100


def banner(title: str, sub: str = "") -> None:
    print()
    print(S.blue("=" * WIDTH))
    print(S.bold(f"  {title}"))
    if sub:
        print(S.dim(f"  {sub}"))
    print(S.blue("=" * WIDTH))


def table(rows: list[list[str]], header: list[str], indent: int = 2) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: " " * indent + "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    print(S.bold(line(header)))
    print(" " * indent + "  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


# ----------------------------------------------------------------------------- stage model
@dataclass
class Stage:
    key: str
    title: str
    what: str                               # one line: what it does / data flow
    cmds: list[list[str]]                   # each is argv after the python executable
    outputs: list[Path] = field(default_factory=list)   # if all exist, the stage counts as done
    check: Callable[[], str | None] = lambda: None      # returns a problem message if inputs are missing
    force_args: list[str] = field(default_factory=list)  # appended to the (single) command under --force
    always: bool = False                    # report/test stages: run every time
    heavy: bool = False                     # long-running: show only key lines unless --verbose
    optional: bool = False                  # a blocked/failed optional stage does not fail the run
    expected: str = ""                      # duration from the saved logs of the author's runs; 'short (not timed)' = not measured
    status: str = "pending"
    seconds: float = 0.0
    note: str = ""


def build_stages(cfg: Config, config_path: Path) -> list[Stage]:
    d, res, gr = cfg.data.dataset_dir, cfg.paths.results_dir, cfg.paths.graphs_dir
    cfgarg = ["--config", str(config_path)]
    cleaned, train, test = d / "cleaned_data.parquet", d / "train.parquet", d / "test.parquet"
    clients = [d / f"client_{c}.parquet" for c in "ABC"]
    fed_ckpt = lambda m: cfg.federated.checkpoint_dir / m / "best.pt"

    def need(*paths: Path) -> Callable[[], str | None]:
        return lambda: next((f"missing input: {p}" for p in paths if not p.exists()), None)

    def need_split_source() -> str | None:
        if train.exists() and test.exists() or cleaned.exists():
            return None
        return f"need {cleaned} (or an existing train/test split)"

    raw = [d / "raw" / f for f in RAW_FILENAMES]
    return [
        Stage("preprocess", "Preprocess raw CICIDS2017", "8 raw parquet files -> merge, drop duplicates, binary label -> cleaned_data.parquet",
              [["preprocessing/preprocess.py"]], [cleaned], need(*raw), expected="short (not timed)"),
        Stage("eda", "Exploratory data analysis", "cleaned_data.parquet -> class-distribution summary + graphs/label_distribution.png",
              [["preprocessing/eda.py"]], [gr / "label_distribution.png"], need(cleaned, *raw), optional=True, expected="short (not timed)"),
        Stage("rf", "Centralized Random Forest (creates train/test split)", "cleaned_data -> 80/20 split -> RF on train -> ONE test evaluation",
              [["-m", "centralized.train_rf", *cfgarg]], [res / "centralized_rf_metrics.json"], need_split_source,
              ["--overwrite"], heavy=True, expected="~2 min"),
        Stage("partition", "Partition train set into clients A/B/C", "train.parquet -> client_A/B/C.parquet (stratified, IID)",
              [["-m", "federated.partition_data", *cfgarg]], clients, need(train), ["--overwrite"], expected="short (not timed)"),
        Stage("optuna", "Optuna search + centralized ANN", "client files (pooled) -> 40-trial search on validation PR-AUC -> retrain best -> ONE test evaluation",
              [["-m", "core.optuna_search", "--final-test", *cfgarg]], [res / "ann_centralized_metrics.json", res / "best_hyperparameters.json"],
              need(*clients), ["--overwrite"], heavy=True, expected="~47 min search + ~1.5 min retrain"),
        Stage("fed-lr", "Federated Logistic Regression", "3 clients x 15 rounds x 3 local epochs, FedAvg, validation each round (no test set)",
              [["-m", "federated.server", "--model", "logistic_regression", *cfgarg]], [res / "federated_logistic_regression_validation_rounds.json"],
              need(*clients), ["--overwrite"], heavy=True, expected="~25 s"),
        Stage("fed-ann", "Federated ANN", "same, with the architecture + local settings from results/best_hyperparameters.json",
              [["-m", "federated.server", "--model", "ann", *cfgarg]], [res / "federated_ann_validation_rounds.json"],
              need(*clients, res / "best_hyperparameters.json"), ["--overwrite"], heavy=True, expected="~2.5 min"),
        Stage("eval-lr", "Final test evaluation: federated LR", "best checkpoint -> threshold from validation -> ONE test evaluation",
              [["-m", "core.evaluate", "--checkpoint", str(fed_ckpt("logistic_regression")), "--name", "lr_federated", "--final-test", *cfgarg]],
              [res / "lr_federated_test_metrics.json"], need(fed_ckpt("logistic_regression")), ["--overwrite"], expected="short (not timed)"),
        Stage("eval-ann", "Final test evaluation: federated ANN", "best checkpoint -> threshold from validation -> ONE test evaluation",
              [["-m", "core.evaluate", "--checkpoint", str(fed_ckpt("ann")), "--name", "ann_federated", "--final-test", *cfgarg]],
              [res / "ann_federated_test_metrics.json"], need(fed_ckpt("ann")), ["--overwrite"], expected="short (not timed)"),
        Stage("reports", "Reports, plots and diagrams", "saved result files -> comparison table, convergence tables, plots, design diagrams (read-only)",
              [["-m", "results.compare_results", *cfgarg], ["-m", "results.print_final_metrics", *cfgarg],
               ["-m", "results.print_convergence", *cfgarg], ["-m", "results.plot_comparison", *cfgarg],
               ["-m", "results.plot_diagrams", *cfgarg]],
              always=True, check=need(res / "centralized_rf_metrics.json", res / "ann_centralized_metrics.json",
                                      res / "lr_federated_test_metrics.json", res / "ann_federated_test_metrics.json",
                                      res / "federated_ann_validation_rounds.json",
                                      res / "federated_logistic_regression_validation_rounds.json"), expected="~10 s"),
        Stage("tests", "Automated test suite", "pytest on small synthetic data (never touches the real dataset or results)",
              [["-m", "pytest", "-q"]], always=True, expected="~4 s (80 tests)"),
    ]


# ----------------------------------------------------------------------------- running commands
def run_cmd(argv: list[str], log, verbose: bool, heavy: bool) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.Popen([sys.executable, "-u", *argv], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    hidden = 0
    for line in proc.stdout:
        log.write(line)
        text = line.rstrip("\n")
        if not text.strip():
            continue
        if heavy and not verbose and not KEY_LINES.search(text):
            hidden += 1
            continue
        print(S.dim("  | ") + text)
    proc.wait()
    log.flush()
    if hidden:
        print(S.dim(f"  | ... {hidden} more lines in the stage log"))
    return proc.returncode


def stage_done(st: Stage) -> bool:
    return bool(st.outputs) and all(p.exists() for p in st.outputs)


def run_stage(st: Stage, args, log_dir: Path) -> None:
    problem = st.check()
    if problem:
        st.status, st.note = ("blocked", problem)
        print(S.yellow(f"  BLOCKED: {problem}"))
        return
    cmds = [list(c) for c in st.cmds]
    if args.force and st.force_args and len(cmds) == 1:
        cmds[0] += st.force_args
    start = time.time()
    with open(log_dir / f"{st.key}.log", "w") as log:
        for argv in cmds:
            shown = " ".join(["python", *[os.path.relpath(a, ROOT) if os.path.isabs(a) and a.startswith(str(ROOT)) else a for a in argv]])
            print(S.cyan(f"  $ {shown}"))
            log.write(f"$ {shown}\n")
            code = run_cmd(argv, log, args.verbose, st.heavy)
            if code != 0:
                st.status, st.note = "failed", f"exit code {code} (see {log_dir / (st.key + '.log')})"
                st.seconds = time.time() - start
                print(S.red(f"  FAILED: {st.note}"))
                return
    st.status, st.seconds = "ok", time.time() - start
    print(S.green(f"  OK in {st.seconds:.1f}s") + S.dim(f"   (log: {rel(log_dir) / (st.key + '.log')})"))


# ----------------------------------------------------------------------------- prelude and summary
def environment_check() -> list[list[str]]:
    rows = []
    for pkg in PACKAGES:
        try:
            rows.append([pkg, md.version(pkg), "ok"])
        except md.PackageNotFoundError:
            rows.append([pkg, "-", "NOT INSTALLED (pip install -r requirements.txt)"])
    return rows


def parquet_rows(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        import pyarrow.parquet as pq
        meta = pq.ParquetFile(path).metadata
        return f"{meta.num_rows:,} rows x {meta.num_columns} cols"
    except Exception:
        return "present"


def data_overview(cfg: Config) -> list[list[str]]:
    d = cfg.data.dataset_dir
    rows = []
    raw_total = 0
    try:
        import pyarrow.parquet as pq
        raw_total = sum(pq.ParquetFile(d / "raw" / f).metadata.num_rows for f in RAW_FILENAMES if (d / "raw" / f).exists())
    except Exception:
        pass
    rows.append(["raw/ (8 files)", f"{raw_total:,} rows" if raw_total else "missing", "CICIDS2017, dhoogla-cleaned"])
    for name, role in [("cleaned_data.parquet", "merged, de-duplicated, binary label"),
                       ("train.parquet", f"{int((1 - cfg.data.test_size) * 100)}% training pool"),
                       ("test.parquet", "held-out test set (used once per model)"),
                       ("client_A.parquet", "organization A"), ("client_B.parquet", "organization B"),
                       ("client_C.parquet", "organization C")]:
        rows.append([name, parquet_rows(d / name), role])
    return rows


def results_table(cfg: Config) -> list[list[str]] | None:
    path = cfg.paths.results_dir / "final_comparison.csv"
    if not path.exists():
        return None
    out = []
    for r in csv.DictReader(open(path)):
        f = lambda k: float(r[k]) if r.get(k) not in (None, "") else None
        out.append([r["model"], f"{f('accuracy') * 100:.2f}%", f"{f('precision') * 100:.2f}%", f"{f('recall') * 100:.2f}%",
                    f"{f('f1') * 100:.2f}%", f"{f('roc_auc'):.4f}", "n/a" if f("pr_auc") is None else f"{f('pr_auc'):.4f}",
                    f"{int(f('false_negatives')):,}", f"{int(f('false_positives')):,}"])
    return out


RESULT_HEADER = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Missed", "False alarms"]


def write_report(path: Path, cfg: Config, stages: list[Stage], total: float, started: datetime, log_dir: Path) -> None:
    md_row = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    lines = [f"# Run report", "",
             f"Generated by `python run_project.py` on {started:%Y-%m-%d %H:%M:%S}. Total {total:.1f}s. "
             f"Full stage logs: `{rel(log_dir)}/`. How the pipeline works: [HOW_IT_WORKS.md](../HOW_IT_WORKS.md).", "",
             "## Stages", "", md_row(["Stage", "Status", "Seconds", "Note"]), md_row(["---"] * 4)]
    for st in stages:
        lines.append(md_row([f"`{st.key}`: {st.title}", "not reached" if st.status == "pending" else st.status, f"{st.seconds:.1f}" if st.seconds else "-", st.note or "-"]))
    res = results_table(cfg)
    if res:
        lines += ["", "## Held-out test results", "", md_row(RESULT_HEADER), md_row(["---"] * len(RESULT_HEADER))]
        lines += [md_row(r) for r in res]
        lines += ["", "ANN / LR thresholds were selected on validation data (max attack-F1); the Random Forest uses 0.5. "
                  "The RF's PR-AUC was not recorded by its original evaluation."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- main
def main() -> int:
    global S
    ap = argparse.ArgumentParser(description="Run the whole project with one command.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--only", default="", help="comma-separated stage keys to run (see --list)")
    ap.add_argument("--skip", default="", help="comma-separated stage keys to skip")
    ap.add_argument("--skip-tests", action="store_true", help="do not run pytest")
    ap.add_argument("--force", action="store_true", help="redo stages even if their outputs exist (re-evaluates the test set!)")
    ap.add_argument("--yes", action="store_true", help="confirm --force without prompting")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    ap.add_argument("--verbose", action="store_true", help="show every output line of long training stages")
    ap.add_argument("--dry-run", action="store_true", help="show the plan only")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    S = Style(sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ)

    cfg = load_config(args.config)
    stages = build_stages(cfg, args.config.resolve())
    keys = [s.key for s in stages]
    only = {k for k in args.only.split(",") if k}
    skip = {k for k in args.skip.split(",") if k}
    unknown = (only | skip) - set(keys)
    if unknown:
        print(f"Unknown stage(s): {sorted(unknown)}. Valid: {keys}")
        return 2
    if args.skip_tests:
        skip.add("tests")

    if args.list:
        table([[s.key, s.title, s.expected or "-"] for s in stages], ["key", "stage", "typical time"])
        return 0

    started = datetime.now()
    t0 = time.time()
    banner("Privacy-Preserving Federated Learning for Malware Detection", "one-command pipeline run")
    print(f"  {platform.platform()}  |  Python {platform.python_version()}  |  config: {args.config.resolve()}")
    print(S.dim("  Data flow: raw files -> cleaned data -> train/test split -> clients A/B/C -> "
                "[RF | Optuna ANN | federated LR | federated ANN] -> validation-only tuning -> one test evaluation each -> reports"))

    banner("1. Environment")
    env = environment_check()
    table(env, ["package", "version", "status"])
    missing = [r[0] for r in env if r[2] != "ok"]
    if missing:
        print(S.red(f"\n  Missing packages: {missing}. Activate the virtual environment and run: pip install -r requirements.txt"))
        return 2

    banner("2. Data on disk")
    table(data_overview(cfg), ["file", "size", "role"])

    banner("3. Plan")
    selected = [s for s in stages if (not only or s.key in only) and s.key not in skip]
    plan = []
    for s in stages:
        if s not in selected:
            action = "not selected"
        elif s.always:
            action = "run (every time)"
        elif args.force:
            action = "RE-RUN (--force)"
        elif stage_done(s):
            action = "skip (outputs exist)"
        else:
            action = "run (outputs missing)"
        plan.append([s.key, s.title, action, s.expected or "-"])
    table(plan, ["key", "stage", "action", "typical time"])
    if args.force:
        print(S.yellow("\n  --force re-trains and RE-EVALUATES models on the held-out test set. The project's protocol is one test "
                       "evaluation per model; use this only deliberately (see HOW_IT_WORKS.md)."))
        if not args.dry_run and not args.yes:
            if not sys.stdin.isatty() or input("  Type 'yes' to continue: ").strip().lower() != "yes":
                print("  Aborted. Re-run with --yes to confirm.")
                return 2
    if args.dry_run:
        print(S.dim("\n  Dry run: nothing executed."))
        return 0

    log_dir = ROOT / "logs" / f"run_{started:%Y%m%d_%H%M%S}"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for n, st in enumerate(stages, 1):
        if st not in selected:
            st.status = "not selected"
            continue
        if not st.always and not args.force and stage_done(st):
            st.status, st.note = "skipped", "outputs already exist"
            print(S.dim(f"\n  [{n}/{len(stages)}] {st.title}: skipped (outputs exist)"))
            continue
        banner(f"[{n}/{len(stages)}] {st.title}", st.what)
        run_stage(st, args, log_dir)
        if st.status == "failed" and not st.optional:
            failed = True
            if not args.keep_going:
                break
        if st.status == "blocked" and not st.optional and not st.always:
            failed = True
            if not args.keep_going:
                break

    total = time.time() - t0
    banner("Summary")
    colors = {"ok": S.green, "skipped": S.dim, "failed": S.red, "blocked": S.yellow, "not selected": S.dim, "pending": S.dim}
    label = lambda st: "not reached" if st == "pending" else st
    table([[s.key, colors.get(s.status, str)(label(s.status).upper()) + " " * (12 - len(label(s.status))), f"{s.seconds:.1f}s" if s.seconds else "-", s.note or ""]
           for s in stages], ["stage", "status", "time", "note"])
    res = results_table(cfg)
    if res:
        print()
        print(S.bold("  Held-out test results (446,362 flows; each model evaluated once)"))
        table(res, RESULT_HEADER)
        print(S.dim("  ANN/LR thresholds chosen on validation (max attack-F1); RF uses 0.5; RF PR-AUC was not recorded."))
    report = cfg.paths.results_dir / "run_report.md"
    write_report(report, cfg, stages, total, started, log_dir)
    print(f"\n  Total {total:.1f}s   |   report: {rel(report)}   |   logs: {rel(log_dir)}/")
    print(S.red("  RESULT: FAILED") if failed else S.green("  RESULT: OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
