"""Apply the exact reviewed patch to a pinned local PhysicsNeMo source checkout."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ARCHIVE_SHA256 = "b480c3603da558046f3ef55c7162bed13e9b60b14b07f2404ef14b9874764c30"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def matches(checkout, files):
    for name, expected in files.items():
        path = checkout / name
        if not path.is_file():
            return False
        content = path.read_bytes()
        if sha256(content) != expected and sha256(content.replace(b"\r\n", b"\n")) != expected:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Path to the PhysicsNeMo Git checkout")
    root = parser.parse_args().checkout.resolve()
    archive = Path(__file__).with_name("MezSolver_public.zip")
    if sha256(archive.read_bytes()) != ARCHIVE_SHA256:
        sys.exit("Review archive SHA-256 mismatch")
    with ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        patch = bundle.read("partitioned-mesh-displacement.patch")
    if sha256(patch) != manifest["patch_sha256"]:
        sys.exit("Patch SHA-256 mismatch")
    git = ["git", "-C", str(root), "-c", "core.autocrlf=false"]
    top = Path(subprocess.check_output(git + ["rev-parse", "--show-toplevel"], text=True).strip())
    if top.resolve() != root:
        sys.exit("Specify the repository root")
    if matches(root, manifest["source_files"]):
        print("All 46 reviewed files already match this checkout.")
        return
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if head != manifest["base"]:
        sys.exit("Use a separate checkout at " + manifest["base"] + " before applying the patch")
    subprocess.run(git + ["apply", "--check", "-"], input=patch, check=True)
    subprocess.run(git + ["apply", "-"], input=patch, check=True)
    if not matches(root, manifest["source_files"]):
        sys.exit("Working-tree files differ from the reviewed bundle after patch application")
    print("Applied and verified the 46 reviewed source files.")


if __name__ == "__main__":
    main()

