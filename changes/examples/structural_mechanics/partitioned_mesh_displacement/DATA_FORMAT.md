# Prepared displacement data

Run `python make_synthetic_data.py --output ./demo_input` to generate a
complete working dataset. The loader reads the same layout for a finite
element model whose mesh is shared by multiple simulation cases. Geometry
and memberships describe the initial configuration. Response fields and
activation masks are indexed by simulation case and physical time.

## Manifest

Supply a JSON file with `format: "partitioned_mesh_displacement_v1"`.
Relative paths below are resolved against the manifest's directory.

| Key | Contents |
| --- | --- |
| `dataset_dir` | Directory containing `cases/`, `training_masks.h5` and, for wheel supervision, `element_activity.h5`. |
| `mesh_cache`, `mesh_sha256` | NumPy NPZ with the ordered mesh arrays below and the SHA-256 of its complete file. |
| `sdf_cache`, `sdf_sha256` | PyTorch tensor dictionary and its SHA-256. |
| `case_ids`, `speed_kmh`, `times` | Case names, impact speed in km/h and an increasing time grid in seconds per case, all in the same order. Each grid starts at zero. The dense time grid has at least three frames per case for one-step supervision. |
| `splits` | Disjoint `train`, `val` and `test` lists. Their union contains each case exactly once. |
| `normalization` | `mean` and positive `std` for the response channels, scalar `speed_mean` and positive `speed_std`, and `fit_cases` listing the training split. Store response means and standard deviations in mm. |
| `dataset_files` (optional) | Map of filenames relative to `dataset_dir` to SHA-256 values. |
| `material_weights`, `material_weights_sha256` (optional) | JSON material calibration and its digest for weighted training. |
| `stage1_validation_panel` (optional) | `patches`, `node_count`, `node_rows` and `node_rows_sha256`. The NPY file holds strictly increasing zero-based global node rows. Its patch membership must equal the supplied list. The fixed-node score is active when the selected validation patch IDs match this list. To reproduce it, pass the same patch IDs as a JSON array to `--val-patch-ids-file`. |
| `seam_edges`, `seam_edges_sha256` (optional) | Catalog of supported shell edges crossing a partition, and its file digest. Used by the Stage 2 and Stage 3 seam losses. |

Use `temporal_reference_time` for new datasets. This finite positive scalar
has the same units as `times` and is shared by every case, split and inference
query. Set it from a known physical scale or fit it on the training split.
When this field is absent, new training uses the median end time of training
cases only. Checkpoints produced before this convention restore their historical
per-case duration. To query an archived prefix, supply its original full case
duration. An archived model cannot distinguish time grids differing solely by
a scale factor when the inputs and normalized query times match.
Stage 1 uses its own observed-step time features.

## Mesh and graph indices

Let `N` be the number of nodes and `P` the number of nonempty node regions.
Supply compact region IDs from 0 through `P-1` and a single region owner per
node. Original material layers sharing a node can retain multiple property
and topology memberships. All CSR pointers start at zero, end at the length
of their index array, and have one entry more than the number of rows.

| NPZ fields | Shape and meaning |
| --- | --- |
| `xyz`, `node_ids`, `scale` | Reference coordinates `[N,3]` in mm, external solver node identifiers `[N]`, and a positive scalar length in mm for geometric edges. Case arrays follow this same node order. |
| `node_patch`, `core_ptr`, `core_indices` | Compact node owner `[N]` and a patch-to-node CSR with one row per patch. Every global node row appears exactly once in the cores. |
| `edge_ptr`, `neighbors`, `edge_flags` | Global node adjacency CSR. Store both directions of each undirected physical edge. Each directed edge has a bit mask with the three source edge-family flags in its low three bits. The model receives the normalized relative position, its length and the three flags. |
| `region_ptr`, `region_neighbors` | Region adjacency CSR used to expand context for inter-region exchange. Recompute it after changing node owners. |
| `owner_region`, `core_owner_region` | Original topology owner for each node and representative topology owner for each patch. Use zero-based topology IDs and `-1` for unassigned. The loader shifts these IDs for the model's categorical embeddings. |
| `property_ptr`, `property_indices`, `property_region_count` | Node-to-property membership CSR and vocabulary size. Named categories are encoded as 1 through the vocabulary size. Empty rows use category 0 in the model. |
| `topology_ptr`, `topology_indices`, `topology_region_count` | Node-to-topology membership CSR with the same convention. |
| `pool_coordinate`, `wall` | One real local pooling coordinate per node and one Boolean flag identifying wall nodes. |
| `static`, `patch_features` | Reference geometry, family and normal descriptors of shape `[N,47]` and `[P,45]`. |

The SDF dictionary stores `node_features` of shape `[N,12]` and
`core_features` of shape `[P,24]`. Concatenating them with the reference
features gives 59 node features. The 45 region features, one normalized
speed and 24 SDF features give 70 region features. Keep the feature order,
units, geometric preprocessing and vocabulary consistent when reusing weights.

## Response and activity

Each `cases/<case>.h5` exposes `fields` of shape `[T,N,C]` with U1, U2 and U3
in millimetres in the first three columns. A node label is valid at a given
frame if `training_masks.h5` marks it in the Boolean array
`cases/<case>/valid_displacement_training` of shape `[T,N]`. Finite
initial displacements can serve as observations at frame zero. Training
supervises positive-time frames. One-step forecasting reads two consecutive
observations and the known time intervals to forecast the next frame.

## Material priorities and wheel groups

Material JSON uses `format: "mesh_material_weights_v1"`, `mesh_sha256`,
a unique `material_types` array, a zero-based `region_type_index` for
each topology region, positive `type_weights` keyed by material name and
`unassigned_weight: 1.0`. Use `material_weight_scope:
"per_node_topology_owner"` when a geometric patch spans multiple materials.
If the file contains `calibration.fit_cases`, use training cases for fitting.
The original material-table format identifier is supported.

For wheel-aware runs, pass a wheel NPZ to `--wheel-groups`. Set
`format: "wheel_shell_groups_v1"`, `mesh_sha256` and `node_ids_sha256`
(the SHA-256 of the contiguous solver ID array in its original dtype).
The loader accepts an optional string vector `wheel_names`. Its defaults
are `front_left`, `front_right`, `rear_left` and `rear_right`. Each wheel
provides the arrays below, prefixed with its name and an underscore.

- `parts` lists its external part IDs. `rows`, `tire_rows` and `rim_rows`
  are sorted, distinct zero-based global node rows. `node_ids` records the
  external solver IDs at `rows`. `patches` lists their compact owner patches.
- `edges[E,2]` stores distinct ascending endpoint pairs in global node rows.
  `edge_shells[E,K]` records incident shell element rows, padded with `-1`.
  An edge remains active while at least one incident shell remains active.
- `tire_edge`, `rim_edge`, `attachment` and `interface` are Boolean arrays
  of length `E`. An attachment edge has exactly one endpoint in `tire_rows`.
  The training interface comprises attachment edges and edges crossing a
  compact patch boundary. The validation attachment metric uses attachment
  edges only.
- `tire_triangles[F,3]` and `tire_elements[F]` identify the reference
  triangles and their source shell element rows. The wheel validation score
  uses active triangles and valid tire nodes.

Wheel runs also require `element_activity.h5` with
`cases/<case>/shell/deleted_indptr` (length `T+1`) and
`cases/<case>/shell/deleted_element_rows`. Each time slice gives the
shell element rows deleted at that frame. These row IDs use the same ordering
as `edge_shells` and `tire_elements`.

## Optional seam supervision in Stages 2 and 3

Use the seam builder in the example README or provide an NPZ catalog bound to
the current mesh by its SHA-256. The catalog contains global node-row pairs
(`edges`), compact region pairs (`patch_pairs`), a region-pair lookup per
edge (`edge_pair`), and supporting shell rows in CSR format (`support_ptr`,
`support_shells`). The builder uses the reference shell topology mapping,
mesh triangles and their source shell rows to identify true perimeter edges.
The synthetic generator writes both the mapping and the completed catalog.

Temporal training evaluates edges with two valid endpoints and at least one
live supporting shell at the target frame. Displacement differences are in
millimeters. A gradient is a displacement difference divided by the reference
edge length in millimeters. The same shell activity HDF5 used for wheel data
is also needed when enabling seam supervision alone.
