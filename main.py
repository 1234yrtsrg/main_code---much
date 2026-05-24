import argparse
import os
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from config import Config


CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH_DEFAULT = CURRENT_DIR / "data" / "Raman_spectroscopy_data_preprocessed.csv"
SINGLE_OUTPUT_DEFAULT = CURRENT_DIR / "nestedcv_oof_predictions.xlsx"
SWEEP_OUTPUT_DEFAULT = CURRENT_DIR / "shared_max_features_sweep_summary.xlsx"
SWEEP_RUN_DIR_DEFAULT = CURRENT_DIR / "shared_max_features_runs"
TABPFN_CACHE_DEFAULT = CURRENT_DIR / ".cache" / "tabpfn"
TARGET_METALS = ["118Sn (KED)", "209Bi (KED)"]


def parse_gpu_ids(gpu_ids_text):
    gpu_ids = []
    for token in str(gpu_ids_text).split(","):
        token = token.strip()
        if not token:
            continue
        gpu_ids.append(int(token))
    return gpu_ids


def configure_runtime(gpu_id=None):
    os.environ["PYTHONWARNINGS"] = "ignore"
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", TABPFN_CACHE_DEFAULT))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)


def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="^Objective did not converge.*")
    warnings.filterwarnings("ignore", category=UserWarning, message="Using a target size .*different to the input size.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")
    warnings.filterwarnings("ignore", message=".*A worker stopped while some jobs were given to the executor.*")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sweep", "single", "worker"],
        default="sweep",
        help="sweep: SHARED_MAX_FEATURES sweep on multiple GPUs; single: run one value; worker: internal subprocess mode."
    )
    parser.add_argument(
        "--shared-max-features",
        type=int,
        default=None,
        help="Single value used by single/worker mode."
    )
    parser.add_argument(
        "--sweep-start",
        type=int,
        default=Config.SHARED_MAX_FEATURES_START,
        help="Inclusive lower bound for SHARED_MAX_FEATURES sweep."
    )
    parser.add_argument(
        "--sweep-end",
        type=int,
        default=Config.SHARED_MAX_FEATURES_END,
        help="Inclusive upper bound for SHARED_MAX_FEATURES sweep."
    )
    parser.add_argument(
        "--sweep-step",
        type=int,
        default=Config.SHARED_MAX_FEATURES_STEP,
        help="Step size for SHARED_MAX_FEATURES sweep."
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=",".join(str(gpu_id) for gpu_id in Config.SWEEP_GPU_IDS),
        help="Comma-separated GPU ids used by sweep mode, for example: 0,1,2,3"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=Config.SKLEARN_N_JOBS,
        help="Shared n_jobs value for sklearn-based CV/search calls."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_PATH_DEFAULT),
        help="Input CSV path."
    )
    parser.add_argument(
        "--single-output",
        type=str,
        default=str(SINGLE_OUTPUT_DEFAULT),
        help="Detailed Excel output path for single mode."
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=str(SWEEP_OUTPUT_DEFAULT),
        help="Final Excel summary path for sweep mode."
    )
    parser.add_argument(
        "--run-output-dir",
        type=str,
        default=str(SWEEP_RUN_DIR_DEFAULT),
        help="Directory used to store sweep partial summaries and logs."
    )
    parser.add_argument(
        "--worker-summary-path",
        type=str,
        default=None,
        help="Internal CSV output path used by worker mode."
    )
    parser.add_argument(
        "--worker-gpu-id",
        type=int,
        default=None,
        help="Internal GPU id used by worker mode."
    )

    args = parser.parse_args()

    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be 0.")

    if args.mode in {"single", "worker"}:
        if args.shared_max_features is None:
            args.shared_max_features = Config.SHARED_MAX_FEATURES
        if args.shared_max_features <= 0:
            parser.error("--shared-max-features must be a positive integer.")

    if args.mode == "worker" and not args.worker_summary_path:
        parser.error("--worker-summary-path is required in worker mode.")

    if args.mode == "sweep":
        if args.sweep_start <= 0 or args.sweep_end <= 0 or args.sweep_step <= 0:
            parser.error("--sweep-start/--sweep-end/--sweep-step must all be positive integers.")
        if args.sweep_end < args.sweep_start:
            parser.error("--sweep-end must be greater than or equal to --sweep-start.")
        try:
            args.gpu_ids = parse_gpu_ids(args.gpu_ids)
        except ValueError as exc:
            parser.error(f"Illegal --gpu-ids value: {exc}")
        if not args.gpu_ids:
            parser.error("--gpu-ids must contain at least one GPU id.")
    else:
        args.gpu_ids = []

    return args


def run_single_training(data_path, output_path, shared_max_features, n_jobs, export_detailed_report, gpu_id=None):
    configure_runtime(gpu_id=gpu_id)
    suppress_warnings()

    Config.SHARED_MAX_FEATURES = shared_max_features
    Config.SKLEARN_N_JOBS = n_jobs

    from pipeline.orchestrator import NestedCVOrchestrator

    print(f"[*] Data Source: {data_path}", flush=True)
    print(f"[*] SHARED_MAX_FEATURES: {shared_max_features}", flush=True)
    print(f"[*] SKLEARN_N_JOBS: {Config.SKLEARN_N_JOBS}", flush=True)
    if gpu_id is not None:
        print(f"[*] Worker GPU: {gpu_id}", flush=True)
    if output_path:
        print(f"[*] Output Target: {output_path}", flush=True)

    runner = NestedCVOrchestrator(
        data_path=str(data_path),
        output_path=str(output_path) if output_path else None,
        target_metals=TARGET_METALS
    )
    return runner.run(export_detailed_report=export_detailed_report)


def run_worker(args):
    summary_path = Path(args.worker_summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_single_training(
        data_path=Path(args.data_path),
        output_path=None,
        shared_max_features=args.shared_max_features,
        n_jobs=args.n_jobs,
        export_detailed_report=False,
        gpu_id=args.worker_gpu_id
    )

    summary_df = result["summary_df"].copy()
    summary_df.insert(0, "SharedMaxFeatures", args.shared_max_features)
    summary_df.to_csv(summary_path, index=False)
    print(f"[*] Worker summary saved to: {summary_path}", flush=True)


def run_gpu_queue(script_path, data_path, shared_values, gpu_id, n_jobs, run_output_dir):
    run_output_dir = Path(run_output_dir)
    summary_dir = run_output_dir / "partials"
    log_dir = run_output_dir / "logs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    completed_files = []
    for shared_value in shared_values:
        summary_path = summary_dir / f"summary_{shared_value:03d}.csv"
        log_path = log_dir / f"shared_{shared_value:03d}_gpu{gpu_id}.log"

        cmd = [
            sys.executable,
            str(script_path),
            "--mode", "worker",
            "--shared-max-features", str(shared_value),
            "--data-path", str(data_path),
            "--n-jobs", str(n_jobs),
            "--worker-summary-path", str(summary_path),
            "--worker-gpu-id", str(gpu_id)
        ]

        print(f"[*] GPU {gpu_id} -> SHARED_MAX_FEATURES={shared_value}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                cmd,
                cwd=str(script_path.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True
            )

        print(f"[OK] GPU {gpu_id} finished SHARED_MAX_FEATURES={shared_value}", flush=True)
        completed_files.append(summary_path)

    return completed_files


def is_summary_csv_complete(summary_path, shared_value):
    import pandas as pd

    summary_path = Path(summary_path)
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return False

    try:
        summary_df = pd.read_csv(summary_path)
    except Exception:
        return False

    if summary_df.empty:
        return False

    required_cols = {"SharedMaxFeatures", "Target", "Model"}
    if not required_cols.issubset(summary_df.columns):
        return False

    try:
        saved_values = set(summary_df["SharedMaxFeatures"].dropna().astype(int).unique())
    except (TypeError, ValueError):
        return False

    return saved_values == {int(shared_value)}


def export_sweep_summary(summary_output, run_output_dir, shared_values=None):
    import pandas as pd
    from pipeline.reporter import ResultReporter

    run_output_dir = Path(run_output_dir)
    summary_dir = run_output_dir / "partials"

    if shared_values is None:
        summary_files = sorted(summary_dir.glob("summary_*.csv"))
    else:
        summary_files = [summary_dir / f"summary_{shared_value:03d}.csv" for shared_value in shared_values]
        missing_files = [summary_file for summary_file in summary_files if not summary_file.exists()]
        if missing_files:
            missing_names = ", ".join(summary_file.name for summary_file in missing_files)
            raise RuntimeError(f"Missing summary CSV files before export: {missing_names}")

    if not summary_files:
        raise RuntimeError(f"No summary CSV files were produced under: {summary_dir}")

    summary_frames = [pd.read_csv(summary_file) for summary_file in summary_files]
    sweep_summary_df = pd.concat(summary_frames, ignore_index=True)
    sweep_summary_df = sweep_summary_df.sort_values(["Target", "SharedMaxFeatures", "Model"])

    combined_csv_path = run_output_dir / "shared_max_features_sweep_summary.csv"
    sweep_summary_df.to_csv(combined_csv_path, index=False)
    ResultReporter.export_sweep_summary_to_excel(str(summary_output), sweep_summary_df)
    print(f"[*] Combined CSV summary: {combined_csv_path}", flush=True)


def get_missing_shared_values(summary_dir, shared_values):
    missing_values = []
    completed_values = []

    for shared_value in shared_values:
        summary_path = summary_dir / f"summary_{shared_value:03d}.csv"
        if is_summary_csv_complete(summary_path, shared_value):
            completed_values.append(shared_value)
        else:
            missing_values.append(shared_value)

    return completed_values, missing_values


def run_sweep(args):
    shared_values = list(range(args.sweep_start, args.sweep_end + 1, args.sweep_step))
    if not shared_values:
        raise RuntimeError("No SHARED_MAX_FEATURES values were generated for the sweep.")

    run_output_dir = Path(args.run_output_dir)
    summary_dir = run_output_dir / "partials"
    log_dir = run_output_dir / "logs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Sweep Range: {shared_values[0]} -> {shared_values[-1]} (step={args.sweep_step})", flush=True)
    print(f"[*] GPUs: {args.gpu_ids}", flush=True)
    print(f"[*] Summary Output: {args.summary_output}", flush=True)
    print(f"[*] Sweep Run Dir: {args.run_output_dir}", flush=True)

    completed_values, missing_values = get_missing_shared_values(summary_dir, shared_values)
    print(f"[*] Completed summaries detected: {len(completed_values)}/{len(shared_values)}", flush=True)

    if not missing_values:
        print("[*] All requested summaries already exist. Exporting combined summary only.", flush=True)
        export_sweep_summary(args.summary_output, args.run_output_dir, shared_values=shared_values)
        return

    print(f"[*] Missing or incomplete SHARED_MAX_FEATURES values: {missing_values}", flush=True)

    assignments = {
        gpu_id: missing_values[offset::len(args.gpu_ids)]
        for offset, gpu_id in enumerate(args.gpu_ids)
        if missing_values[offset::len(args.gpu_ids)]
    }

    script_path = CURRENT_DIR / "main.py"
    completed_files = []
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [
            executor.submit(
                run_gpu_queue,
                script_path,
                Path(args.data_path),
                values,
                gpu_id,
                args.n_jobs,
                args.run_output_dir
            )
            for gpu_id, values in assignments.items()
        ]

        for future in as_completed(futures):
            completed_files.extend(future.result())

    _, remaining_missing_values = get_missing_shared_values(summary_dir, shared_values)
    if remaining_missing_values:
        raise RuntimeError(
            f"Still missing summary CSV files for SHARED_MAX_FEATURES={remaining_missing_values}."
        )

    export_sweep_summary(args.summary_output, args.run_output_dir, shared_values=shared_values)


def run_single(args):
    run_single_training(
        data_path=Path(args.data_path),
        output_path=Path(args.single_output),
        shared_max_features=args.shared_max_features,
        n_jobs=args.n_jobs,
        export_detailed_report=True,
        gpu_id=None
    )


if __name__ == "__main__":
    cli_args = parse_args()

    if cli_args.mode == "worker":
        run_worker(cli_args)
    elif cli_args.mode == "single":
        run_single(cli_args)
    else:
        run_sweep(cli_args)
