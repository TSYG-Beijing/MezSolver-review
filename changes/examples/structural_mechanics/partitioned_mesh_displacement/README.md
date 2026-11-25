# Three-stage displacement prediction on partitioned meshes

This example learns three displacement components on a fixed finite-element
mesh. A graph encoder uses surrounding mesh nodes to give each region local
context. Only the region's core nodes contribute to its pooled latent tokens.
A node query attends to those tokens and a region-conditioned residual decoder
reconstructs the displacement. A second model uses two observed latent states
and the physical time intervals to forecast the next state.

This example complements the automotive crash recipes with a spatial
autoencoder and an auxiliary two-observation prediction task. The displacement
reader accepts prepared meshes with mixed elements and overlapping physical
memberships. See the [existing automotive crash example](../crash/README.md)
and the [input data specification](DATA_FORMAT.md).

## Training order and checkpoint handoff

| Order | Entry point | Input | Parameters updated |
| --- | --- | --- | --- |
| 1. Spatial training | `train_stage_1.py` | Prepared mesh and training cases | Spatial autoencoder and auxiliary one-step predictor |
| 2. Trajectory prediction | `train_stage_2.py` | Stage 1 validation-selected `best.pt` | Temporal state-space model |
| 3. Joint refinement | `train_stage_3.py` | Stage 2 validation-selected `best.pt` | Temporal model and spatial decoder by default |

Stage 1 reconstructs each observed field and uses two observed latent states to
supervise a one-step forecast. Stage 2 loads the spatial encoder and decoder,
freezes them and initializes a separate trajectory model. Given the initial
field, physical time grid and known conditions, that model predicts the
trajectory without reading later response fields. Stage 3 inherits both
models from Stage 2 and can refine how trajectory states decode to nodal
displacements. Its default keeps the encoder frozen. A mesh and data split
must remain consistent across all three stages.

## Run all three stages on CPU

From the repository root, install PhysicsNeMo according to its installation
guide and then run the commands below. The generator writes a small analytic
dataset with four annular meshes, independent case splits, and irregular time
steps. It requires an empty destination directory.

```bash
cd examples/structural_mechanics/partitioned_mesh_displacement
python -m pip install -r requirements.txt
python make_synthetic_data.py --output demo_data
python train_stage_1.py --manifest demo_data/manifest.json --output runs/stage1 \
    --device cpu --epochs 1 --steps-per-epoch 2 --patches-per-batch 6 \
    --hidden-dim 32 --coarse-dim 16 --fine-dim 8 --decoder-hidden-dim 32 \
    --region-attention-heads 4 --region-attention-width 32 \
    --region-balance-power 1 --material-weight-power 1 \
    --wheel-groups demo_data/wheels.npz --wheel-batch-fraction 1 \
    --wheel-edge-weight 0.2 --wheel-interface-weight 0.1 \
    --val-patch-ids-file demo_data/patches.json
python train_stage_2.py --manifest demo_data/manifest.json \
    --initial-checkpoint runs/stage1/best.pt --output runs/stage2 \
    --device cpu --epochs 1 --steps-per-epoch 2 --patches-per-batch 6 \
    --frames-per-step 2 --hidden-dim 16 --state-dim 4 \
    --region-attention-heads 2 --region-attention-width 16 --dynamic-region-exchange \
    --wheel-groups demo_data/wheels.npz --wheel-batch-fraction 0.4 \
    --wheel-edge-weight 0.2 --wheel-interface-weight 0.1
python train_stage_3.py --manifest demo_data/manifest.json \
    --initial-checkpoint runs/stage2/best.pt --output runs/stage3 \
    --device cpu --epochs 1 --steps-per-epoch 2 --patches-per-batch 6 \
    --frames-per-step 2 --unfreeze decoder \
    --wheel-groups demo_data/wheels.npz --wheel-batch-fraction 0.4 \
    --wheel-edge-weight 0.2 --wheel-interface-weight 0.1 \
    --seam-edges demo_data/seams.npz --seam-batch-fraction 0.4 \
    --seam-gradient-weight 0.005
```

Each command writes a validation-selected `best.pt` and a resumable `last.pt`.
Choose a different output directory for each stage. The short runs above
exercise the handoff. For Stage 2, the optional `--dynamic-region-exchange`
adds a same-time neighbor update between the first two SSM blocks. Stage 3
inherits its temporal architecture and weights from the Stage 2 checkpoint.

## Evaluate and export

```bash
python evaluate_components.py --manifest demo_data/manifest.json \
    --checkpoint runs/stage1/best.pt --split val --frames 2 3 4 \
    --output runs/stage1/components.json
python evaluate_components.py --manifest demo_data/manifest.json \
    --checkpoint runs/stage3/best.pt --split test --full-mesh \
    --output runs/stage3/test_components.json
python export_models.py --checkpoint runs/stage3/best.pt --output exported_stage3
```

The evaluator pools valid node-frame pairs and reports RMSE in millimeters
and dimensionless relative L2 for U1, U2 and U3. Stage 1 compares
reconstruction with a one-step forecast from the previous two observations.
For Stage 2 or 3 it compares reconstruction with the trajectory predicted
from the initial field. Use the same mesh scope and target frames when
comparing stages. A held-out test split is only evaluated on request.

Each export contains `autoencoder.mdlus` and either `one_step.mdlus`
(Stage 1) or `temporal.mdlus` (Stages 2 and 3). For inference, load with
`physicsnemo.core.Module.from_checkpoint(...)` and use the mesh preprocessing,
physical time grid and normalization statistics supplied with the dataset.

For Stage 2 and Stage 3 the export also writes `trajectory_clock.json`. Pass
its `temporal_reference_time` as the predictor's `time_scale` argument when
querying a full trajectory or its prefix. A null clock identifies historical
per-case normalization and requires the original full case duration at inference.

## Optional fused selective scan on CUDA

The default `parallel` scan uses PyTorch on CPU and GPU. On Linux with a
matching CUDA toolkit and CUDA-enabled PyTorch already installed, install
the optional Mamba backend from this example directory:

```bash
python -m pip install --no-build-isolation -r requirements-mamba.txt
```

Pass `--scan-backend mamba` when creating the Stage 2 model. The training and
evaluation commands bind the kernel to the requested blocks automatically.
To run an exported fused model, call
`displacement.scan_backend.configure_scan_backend(model)` after loading its
`.mdlus` file. Binding is per instance and adds no parameters or buffers.
Historical checkpoints keep the same tensor layout. For execution without
the extension, construct the temporal model with the saved architecture and
`scan_backend="parallel"`, and load the same state dictionary strictly.

## Distributed training with prepared data

Set the four paths below to the prepared assets and an output directory,
then run the distributed trainer. This example's data locations and case
splits are carried by the manifest. Relative paths in the manifest resolve
from its own directory.

```bash
export MANIFEST=/path/to/data/manifest.json
export WHEELS=/path/to/data/wheel_training_groups.npz
export PANEL=/path/to/data/validation_patches.json
export OUTPUT=/path/to/run
torchrun --standalone --nproc-per-node=4 train_stage_1.py --preset wheel-supervised \
    --manifest "$MANIFEST" --wheel-groups "$WHEELS" \
    --val-patch-ids-file "$PANEL" --output "$OUTPUT"
```

For an interrupted run, use the same command, configuration and process
count, adding `--resume "$OUTPUT/last.pt"` for a checkpoint from this recipe.
The `wheel-supervised` preset uses 500 epochs, 256 optimizer steps per epoch, 16 requested
regions per rank, BF16 training and a starting learning rate of 0.0003.
The spatial model uses three mesh message-passing layers, one four-head
region exchange layer and four decoder residual blocks. The training loss
includes reconstructed fields, a next-latent term weighted by 0.1 and a
decoded one-step term weighted by 0.02. Node-count and material weighting use
exponent 1. Wheel batches are sampled with probability 0.4. Interior and
interface wheel-edge displacement differences carry weights 0.2 and 0.1.
The two models are initialized together at the start of Stage 1.

The training-time scalar used for checkpoint selection combines normalized
reconstruction error and the optional wheel metric. Use
`evaluate_components.py` to measure one-step prediction accuracy directly.

Run Stage 2 and Stage 3 in sequence after the preceding stage has produced
its final validation-selected checkpoint. Here `OUTPUT` is the Stage 1 run,
and the next two runs have their own output directories. The 900 and 200
epochs illustrate a full training schedule. Set budgets for your dataset.

```bash
torchrun --standalone --nproc-per-node=4 train_stage_2.py \
    --manifest "$MANIFEST" --initial-checkpoint "$OUTPUT/best.pt" \
    --output runs/stage2 --device cuda --epochs 900 --spatial-layers 2 \
    --temporal-layers 2 --region-attention-heads 4 --region-attention-width 64 \
    --dynamic-region-exchange --wheel-groups "$WHEELS" \
    --wheel-batch-fraction 0.4 --wheel-edge-weight 0.2 --wheel-interface-weight 0.1
torchrun --standalone --nproc-per-node=4 train_stage_3.py \
    --manifest "$MANIFEST" --initial-checkpoint runs/stage2/best.pt \
    --output runs/stage3 --device cuda --epochs 200 --unfreeze decoder \
    --wheel-groups "$WHEELS" --wheel-batch-fraction 0.4 \
    --wheel-edge-weight 0.2 --wheel-interface-weight 0.1
```

Optional seam supervision matches the predicted displacement difference
across a physical edge to the simulated difference. Gradient supervision
accounts for the reference edge length. Use `--seam-weight` for the
displacement difference and `--seam-gradient-weight` for the gradient,
and supply separate wheel and seam sampling fractions. The loss applies to
active shell edges crossing partitions within the same physical shell region.
For a different mesh, build its own catalog before passing `--seam-edges`.

```bash
python -m displacement.temporal_seams --mesh mesh.npz \
    --mapping shell_topology.h5 --output seams.npz
```

## Optional concentric wheel partitions

`wheel_rings.py` takes one wheel's reference points, shell connectivity, and
the point clouds of the two bead rings. It fits the axle and assigns node and
cell bands by their perpendicular distance to that axle. The following command
uses millimeter coordinates and five internal boundaries. Boundaries are
configurable for other wheel geometries.

```bash
python wheel_rings.py --input wheel_reference.npz \
    --boundaries 90 150 208 250 285 --output wheel_bands.npz
```

The output contains one-based ring labels within that wheel. During mesh
preparation, assign each wheel unique geometric IDs, combine them with the
other vehicle regions, and compact node IDs for use as training patches.
The recipe consumes the resulting prepared mesh and wheel edge groups. The
synthetic generator provides a complete working example of the input files.
