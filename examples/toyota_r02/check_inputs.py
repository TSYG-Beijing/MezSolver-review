"""Check the R02 geometry, data identities, case inventory and HDF5 metadata."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_assets(manifest):
    report = read_json(ROOT / "provenance/asset_conversion.json")
    for name, record in report["file_sha256"].items():
        if name.startswith("assets/") or name == "manifest.json":
            require(digest(ROOT / name) == record["sha256"], f"Package digest mismatch: {name}")
    for path_key, hash_key in (("mesh_cache", "mesh_sha256"), ("sdf_cache", "sdf_sha256"),
                               ("material_weights", "material_weights_sha256"), ("seam_edges", "seam_edges_sha256")):
        require(digest(ROOT / manifest[path_key]) == manifest[hash_key], f"Manifest digest mismatch: {path_key}")
    panel = manifest["stage1_validation_panel"]
    rows = np.load(ROOT / panel["node_rows"], allow_pickle=False)
    patches = read_json(ROOT / "assets/validation_patches.json")
    require(len(rows) == panel["node_count"] and patches == panel["patches"], "Validation panel differs")
    with np.load(ROOT / manifest["mesh_cache"], allow_pickle=False) as mesh:
        owner = mesh["node_patch"]
        require(len(owner) == manifest["nodes"] and len(mesh["core_ptr"]) - 1 == manifest["patches"], "Mesh dimensions differ")
        require(np.array_equal(np.unique(owner[rows]), patches), "Validation rows belong to different patches")
    ids = manifest["case_ids"]
    split_ids = [case for split in ("train", "val", "test") for case in manifest["splits"][split]]
    require(len(set(ids)) == len(ids) == len(split_ids) and set(ids) == set(split_ids), "Case split mismatch")
    print(f"Verified {manifest['nodes']:,} nodes, {manifest['patches']:,} regions, {len(rows):,} validation nodes and {len(ids)} cases.")


def check_data(manifest):
    import h5py

    data = ROOT / manifest["dataset_dir"]
    if not data.is_dir():
        raise FileNotFoundError(f"Mount the prepared data at {data} (see README.md)")
    for filename, expected in manifest["dataset_files"].items():
        require(digest(data / filename) == expected, f"Dataset identity mismatch: {filename}")
    ids = manifest["case_ids"]
    with h5py.File(data / "training_masks.h5", "r") as masks, h5py.File(data / "element_activity.h5", "r") as active:
        for case, time in zip(ids, manifest["times"], strict=True):
            with h5py.File(data / "cases" / (case + ".h5"), "r") as sample:
                shape = sample["fields"].shape
                expected = (len(time), manifest["nodes"])
                require(len(shape) == 3 and shape[:2] == expected and shape[2] >= 3, f"Invalid fields: {case} {shape}")
                require(masks[f"cases/{case}/valid_displacement_training"].shape == expected, f"Invalid mask: {case}")
                shell = active[f"cases/{case}/shell"]
                require(shell["deleted_indptr"].shape == (len(time) + 1,) and "deleted_element_rows" in shell,
                        f"Invalid shell activity: {case}")
        case = manifest["splits"]["train"][0]
        with h5py.File(data / "cases" / (case + ".h5"), "r") as sample:
            for start in (0, manifest["nodes"] // 2):
                stop = min(start + 256, manifest["nodes"])
                try:
                    vector = sample["fields"][1, start:stop, :3]
                    valid = masks[f"cases/{case}/valid_displacement_training"][1, start:stop].astype(bool)
                    _ = active[f"cases/{case}/shell/deleted_indptr"][1:3]
                except OSError as exc:
                    raise OSError(f"HDF5 chunk read failed for {case}, frame 1, nodes {start}:{stop} in {data}. Check the data mount or verified local mirror.") from exc
                require(np.isfinite(vector[valid]).all(), f"Nonfinite valid training displacements: {case}")
    print(f"Verified {len(ids)} response headers, mask and shell-activity shapes, and two displacement chunks from {case}. No test labels read.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--assets-only", action="store_true", help="Validate data-independent assets without mounting HDF5")
    args = cli.parse_args()
    manifest = read_json(ROOT / "manifest.json")
    check_assets(manifest)
    if not args.assets_only:
        check_data(manifest)


if __name__ == "__main__":
    main()

