"""Launch the Toyota R02 three-stage experiment with the integrated PhysicsNeMo example."""

import argparse
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
EXAMPLE = Path("examples/structural_mechanics/partitioned_mesh_displacement")


def parse_args():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stage", choices=("all", "1", "2", "3"), default="all")
    cli.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="Processes on this host; CUDA_VISIBLE_DEVICES selects the devices.",
    )
    cli.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Run root; each stage uses its own stage1, stage2 or stage3 directory.",
    )
    cli.add_argument(
        "--physicsnemo-root",
        type=Path,
        help="Patched PhysicsNeMo checkout. Defaults to PHYSICSNEMO_ROOT or the installed checkout.",
    )
    cli.add_argument(
        "--resume",
        action="store_true",
        help="Resume the chosen stage from last.pt. With all, skip completed stages and also start pending ones.",
    )
    cli.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without opening training data, accessing a GPU or writing output.",
    )
    cli.add_argument(
        "--epochs", type=int, nargs=3, default=(500, 900, 200), metavar=("S1", "S2", "S3")
    )
    cli.add_argument("--steps-per-epoch", type=int, default=256)
    cli.add_argument(
        "--response-cache-mb",
        type=int,
        help="Override response cache per dataset instance (MiB); historical values 65536 / 16384 / 16384.",
    )
    cli.add_argument(
        "--validation-cache-mb",
        type=int,
        default=8192,
        help="Validation field cache setting per process in MiB.",
    )
    cli.add_argument("--open-cases", type=int, default=128)
    cli.add_argument(
        "--seam-gradient-weights",
        type=float,
        nargs=2,
        default=(0.005, 0.01),
        metavar=("S2", "S3"),
        help="Zero disables seam supervision in that stage.",
    )
    cli.add_argument(
        "--seam-batch-fraction",
        type=float,
        default=0.3,
        help="Fraction of seam batches when enabled; wheel batches take another 0.4.",
    )
    args = cli.parse_args()
    if min(args.gpus, args.steps_per_epoch, args.open_cases, *args.epochs) < 1:
        cli.error("GPU, epoch, step and open-case counts must be positive")
    if args.validation_cache_mb < 0 or (
        args.response_cache_mb is not None and args.response_cache_mb < 0
    ):
        cli.error("Cache budgets must be nonnegative")
    if any(not math.isfinite(w) or w < 0 for w in args.seam_gradient_weights):
        cli.error("Seam weights must be finite and nonnegative")
    if not 0 <= args.seam_batch_fraction <= 0.6:
        cli.error("Seam sampling must be in [0, 0.6] with the wheel fraction at 0.4")
    if args.stage != "1" and args.seam_batch_fraction == 0:
        active = (
            args.seam_gradient_weights
            if args.stage == "all"
            else (args.seam_gradient_weights[int(args.stage) - 2],)
        )
        if any(active):
            cli.error("Positive seam weights require a positive seam sampling fraction")
    args.output = args.output.expanduser().resolve()
    return args


def checkout_for(args):
    configured = args.physicsnemo_root or os.environ.get("PHYSICSNEMO_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        spec = importlib.util.find_spec("physicsnemo")
        if spec is None or not spec.origin:
            raise ValueError("Set --physicsnemo-root to the patched checkout")
        root = Path(spec.origin).resolve().parents[1]
    for path in [root / EXAMPLE / f"train_stage_{stage}.py" for stage in (1, 2, 3)] + [
        root / "physicsnemo/experimental/models/partitioned_mesh/__init__.py"
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing integrated source at {path}. Apply the review patch first."
            )
    return root


def completed(stage_dir, stage, epochs):
    marker = stage_dir / "completion.json"
    if not marker.is_file():
        return False
    info = json.loads(marker.read_text(encoding="utf-8"))
    if not (
        info.get("status") == "complete"
        and info.get("stage") == stage
        and info.get("epochs") == epochs
        and 1 <= info.get("best_epoch", 0) <= epochs
        and math.isfinite(info.get("best_score", math.inf))
        and info.get("test_set_used") is False
        and (stage_dir / "best.pt").is_file()
    ):
        raise ValueError(
            f"Missing or mismatched completed Stage {stage} handoff in {stage_dir}; check the requested --epochs"
        )
    return True


def stage_command(args, root, stage, resume):
    directory = args.output / f"stage{stage}"
    assets = BUNDLE / "assets"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.gpus}",
        "--max-restarts=0",
        str(root / EXAMPLE / f"train_stage_{stage}.py"),
        "--manifest",
        str(BUNDLE / "manifest.json"),
        "--output",
        str(directory),
        "--device",
        "cuda",
        "--epochs",
        str(args.epochs[stage - 1]),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--seed",
        "42",
        "--wheel-groups",
        str(assets / "wheel_training_groups.npz"),
        "--val-patch-ids-file",
        str(assets / "validation_patches.json"),
        "--validation-cache-mb",
        str(args.validation_cache_mb),
        "--open-cases",
        str(args.open_cases),
    ]
    response_cache = args.response_cache_mb
    if response_cache is None:
        response_cache = 65536 if stage == 1 else 16384
    command.extend(["--response-cache-mb", str(response_cache)])
    if resume:
        command.extend(["--resume", str(directory / "last.pt")])
    elif stage > 1:
        command.extend(["--initial-checkpoint", str(args.output / f"stage{stage - 1}" / "best.pt")])
    if stage == 1:
        return command + ["--preset", "wheel-supervised"]
    command.extend(
        [
            "--patches-per-batch",
            "2",
            "--eval-patches-per-batch",
            "8",
            "--frames-per-step",
            "4",
            "--learning-rate",
            "0.0001" if stage == 2 else "0.00005",
            "--min-learning-rate",
            "0.000001",
            "--region-balance-power",
            "1",
            "--material-weight-power",
            "1",
            "--channel-weights",
            "1",
            "1",
            "1",
            "--wheel-batch-fraction",
            "0.4",
            "--wheel-edge-weight",
            "0.2",
            "--wheel-interface-weight",
            "0.1",
            "--wheel-selection-weight",
            "0.1",
            "--val-max-cases",
            "12",
            "--val-max-frames",
            "5",
            "--val-every",
            "1",
            "--max-graph-nodes",
            "1000000",
            "--max-graph-edges",
            "8000000",
            "--cpu-threads",
            "4",
            "--precision",
            "bf16",
            "--matmul-precision",
            "highest",
            "--fused-adamw",
        ]
    )
    weight = args.seam_gradient_weights[stage - 2]
    if weight:
        command.extend(
            [
                "--seam-edges",
                str(assets / "seam_pairs.npz"),
                "--seam-batch-fraction",
                str(args.seam_batch_fraction),
                "--seam-weight",
                "0",
                "--seam-gradient-weight",
                str(weight),
            ]
        )
    if stage == 2:
        command.extend(
            [
                "--hidden-dim",
                "128",
                "--state-dim",
                "16",
                "--spatial-layers",
                "2",
                "--temporal-layers",
                "2",
                "--region-attention-heads",
                "4",
                "--region-attention-width",
                "64",
                "--dynamic-region-exchange",
                "--scan-backend",
                "parallel",
            ]
        )
    else:
        command.extend(["--unfreeze", "decoder", "--reconstruction-weight", "0.1"])
    return command


def main():
    args = parse_args()
    root = checkout_for(args)
    stages = (1, 2, 3) if args.stage == "all" else (int(args.stage),)
    for stage in stages:
        directory = args.output / f"stage{stage}"
        if directory.exists() and not args.resume:
            raise FileExistsError(
                f"{directory} already exists. Use a new output root or explicitly resume."
            )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root), environment.get("PYTHONPATH")))
    )
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(key, "4")
    if not args.dry_run:
        if os.name != "posix":
            raise ValueError(
                "Use the Linux CUDA training environment for distributed training. --dry-run is portable."
            )
        required = [
            BUNDLE / "manifest.json",
            BUNDLE / "assets/wheel_training_groups.npz",
            BUNDLE / "assets/validation_patches.json",
        ]
        if any(stage > 1 and args.seam_gradient_weights[stage - 2] > 0 for stage in stages):
            required.append(BUNDLE / "assets/seam_pairs.npz")
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Missing input {path}")
        probe = """
import sys
from pathlib import Path
import torch
import physicsnemo.experimental.models.partitioned_mesh as model

if Path(model.__file__).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit("Imported model is outside the selected PhysicsNeMo checkout")
if torch.cuda.device_count() < int(sys.argv[2]):
    raise SystemExit("Fewer visible CUDA devices than --gpus")
print("CUDA devices:", torch.cuda.device_count(), "model:", model.__file__)
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                probe,
                str(root / "physicsnemo/experimental/models/partitioned_mesh/__init__.py"),
                str(args.gpus),
            ],
            check=True,
            env=environment,
            cwd=root / EXAMPLE,
        )
    for stage in stages:
        directory = args.output / f"stage{stage}"
        epochs = args.epochs[stage - 1]
        if args.resume and completed(directory, stage, epochs):
            print(f"Stage {stage} is complete; using {directory / 'best.pt'}", flush=True)
            continue
        resume = args.resume and (directory.exists() or args.stage != "all")
        if resume and not args.dry_run and not (directory / "last.pt").is_file():
            raise FileNotFoundError(f"No resumable checkpoint at {directory / 'last.pt'}")
        if stage > 1 and not resume and not args.dry_run:
            parent = args.output / f"stage{stage - 1}"
            if not completed(parent, stage - 1, args.epochs[stage - 2]):
                raise ValueError(
                    f"Stage {stage} requires a completed parent and its validation-selected best.pt in {parent}"
                )
        command = stage_command(args, root, stage, resume)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=root / EXAMPLE, env=environment)
            if not completed(directory, stage, epochs):
                raise RuntimeError(
                    f"Stage {stage} exited without a verified completion.json and best.pt"
                )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
