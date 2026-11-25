# Apply this candidate

Check out NVIDIA/physicsnemo at `f32f11b3c63b683af3b2af1afce49a105019974c`. Apply this patch to a clean checkout. Each candidate is a complete patch against the same base. Choose a single candidate for the checkout.

```bash
git apply --check /path/to/partitioned-mesh-displacement.patch
git apply /path/to/partitioned-mesh-displacement.patch
```

Install the patched checkout following the upstream installation guide. Run the CPU quickstart in `examples/structural_mechanics/partitioned_mesh_displacement/README.md`. Run model and example tests with `python -m pytest test/experimental/models/partitioned_mesh examples/structural_mechanics/partitioned_mesh_displacement/tests`. The optional fused CUDA regression requires its compatible kernel and GPU.

The manifest lists the changed files and patch digest. The terms and notices accompanying the source are in LICENSE.txt and THIRD_PARTY_NOTICES.md.
