# MezSolver contribution review

This repository stages a proposed PhysicsNeMo contribution for internal review. The release contains a partitioned-mesh spatial autoencoder, a two-observation one-step predictor, and a selective state-space predictor for U1, U2 and U3. The example covers three training stages, component-level evaluation and model export.

- [Reviewed source package](MezSolver_public.zip) contains the patch, changed files, checksum manifest, license and notices. Browse the [model sources](changes/physicsnemo/experimental/models/partitioned_mesh) and [example](changes/examples/structural_mechanics/partitioned_mesh_displacement/README.md) directly. Follow [APPLY.md](APPLY.md) to apply the patch and run the synthetic example.
- [Maintainer review note](MezSolver_Maintainer_Review.docx) describes the models, training sequence, optional CUDA selective-scan backend and CPU verification.

## Release identity

Upstream repository: NVIDIA/physicsnemo. Base commit: `f32f11b3c63b683af3b2af1afce49a105019974c`.

| Artifact | SHA-256 |
| --- | --- |
| MezSolver_public.zip | `b480c3603da558046f3ef55c7162bed13e9b60b14b07f2404ef14b9874764c30` |
| partitioned-mesh-displacement.patch (inside the archive) | `be229ea6aed56339c4ff336892eadc0c4e0e966a0689737c7aaca5c77a7cad42` |

## Items for the upstream pull request

Add a real-data example with a released dataset or documented access path and a reproducible evaluation. Include matched baseline results, preferably GeoTransolver, and report the integrated model's results in the example README and PR description. The current package documents the synthetic smoke run.
