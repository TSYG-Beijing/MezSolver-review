"""Stage Ceph-resident Yaris labels with direct I/O and verify historical SHA-256."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def direct_copy(src, dst):
    subprocess.run(["dd", f"if={src}", f"of={dst}", "iflag=direct", "bs=4M", "status=none"], check=True)
    if src.stat().st_size != dst.stat().st_size:
        raise ValueError(f"Size mismatch for {src.name}")


def aux_check(file, name):
    if name.suffix == ".json":
        return json.loads(file.read_text()) is not None
    if name.suffix != ".h5":
        return True
    with h5py.File(file, "r") as h:
        if name.name.startswith("Toyota_yaris_116"):
            assert h["fields"].shape == (116,31,1488757,5)
            ids = h["case_ids"].asstr()[:]
            speeds = h["speed_kmh"][:]
            assert len(set(ids)) == 116 and np.isfinite(speeds).all() and speeds.min()==40 and speeds.max()==60
        elif name.name.startswith("geometry_cores"):
            assert h["core_indices"].shape == (1488757,)
            assert h["core_ptr"][-1] == 1488757
        else:
            assert len(h.keys()) > 0


def publish_file(root, dest, name, expected):
    output = dest / name
    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(exist_ok=True, parents=True)
    if output.exists() and expected and checksum(output) == expected:
        return {"file": str(name), "sha256": expected, "existing": True}
    if name == Path("cases/F005.h5") and partial.exists() and checksum(partial) == expected:
        got = expected
    else:
        direct_copy(root / name, partial)
        got = checksum(partial)
    if expected and got != expected:
        raise ValueError(f"{name} computed {got}, expected {expected}")
    if name.parts[0] == "cases":
        if expected is None:
            raise ValueError(f"No historical checksum for {name}")
    else:
        aux_check(partial,name)
    os.replace(partial, output)
    return {"file": str(name), "sha256": got, "historical": bool(expected)}


def log_status(file, status):
    pending = file.with_suffix(".tmp")
    pending.write_text(json.dumps(status, indent=2))
    os.replace(pending, file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("destination",type=Path)
    p.add_argument("progress",type=Path)
    p.add_argument("--workers",type=int,default=12)
    args = p.parse_args()
    source = args.source.resolve()
    dst = args.destination.resolve()
    if source.is_relative_to(dst) or dst.is_relative_to(source):
        p.error("Choose a mirror destination separate from the original data tree")
    if args.progress.resolve().is_relative_to(source):
        p.error("Keep the progress file outside the original data tree")
    if args.workers < 1:
        p.error("--workers must be positive")
    dst.mkdir(exist_ok=True,parents=True)
    lock = (dst / ".stage.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    direct_copy(args.source/"SHA256SUMS", dst/"SHA256SUMS.partial")
    entries = {}
    for line in (dst/"SHA256SUMS.partial").read_text().splitlines():
        digest, name = line.split(None,1)
        entries[Path(name.lstrip("*"))] = digest
    os.replace(dst/"SHA256SUMS.partial",dst/"SHA256SUMS")
    cases = sorted((args.source/"cases").glob("F*.h5"))
    assert len(cases) == 116 and all(Path("cases")/f.name in entries for f in cases)
    required_json = ["metadata.json","dataset_manifest.json","source_index.json","completion_summary.json","region_validation.json","data_quality_report.json"]
    aux = [x.name for x in args.source.glob("*.h5")] + required_json
    names = [Path(n) for n in aux]+[Path("cases")/f.name for f in cases]
    status = {"source":str(args.source),"destination":str(dst),"started":time.time(),"total":len(names),"cases_ok":0,"aux_ok":0,"state":"running","failures":[],"verified":[]}
    log_status(args.progress,status)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        tasks={ex.submit(publish_file,args.source,dst,n,entries.get(n)):n for n in names}
        for future in concurrent.futures.as_completed(tasks):
            name=tasks[future]
            try:
                info=future.result()
                status["verified"].append(info)
                status["cases_ok" if name.parts[0]=="cases" else "aux_ok"]+=1
            except Exception as err:
                status["failures"].append({"file":str(name),"error":str(err)})
            status["updated"]=time.time()
            log_status(args.progress,status)
    status["state"]="complete" if not status["failures"] else "failed"
    log_status(args.progress,status)
    print(status["state"],status["cases_ok"],status["aux_ok"],status["failures"],flush=True)
    return int(bool(status["failures"]))


if __name__=="__main__":
    raise SystemExit(main())
