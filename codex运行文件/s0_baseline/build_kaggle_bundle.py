"""Build a self-contained Kaggle script; datasets and weights are attached in the UI."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
SOURCES = ("s0_model.py", "s0_data.py", "s0_merged_validation.py", "train_s0.py", "requirements.txt",
           "protocol/s0_selection.json", "protocol/s0_selection_ids.csv",
           "protocol/merged_validation_reference.json",
           "protocol/protocol.json", "protocol/data_audit.json")


def build(output: Path | None = None) -> Path:
    output = output or HERE / "kaggle_bundle"
    output.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (HERE / name).read_bytes())
    script = '''# Generated S0 source bundle. Attach PV-512 and official DINOv3-SAT weights.
import base64, io, pathlib, subprocess, sys, zipfile
root = pathlib.Path("/kaggle/working/s0_source")
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(root)
subprocess.run([sys.executable, str(root / "train_s0.py"), *sys.argv[1:]], check=True)
'''.replace("PAYLOAD", repr(base64.b64encode(payload.getvalue()).decode("ascii")))
    script_path = output / "run-s0-baseline.py"
    script_path.write_text(script, encoding="utf-8")
    metadata = {"id": "zh0416/pv-dinov3sat-s0-baseline", "title": "PV DINOv3-SAT S0 Baseline",
                "code_file": script_path.name, "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": [], "model_sources": [],
                "kernel_sources": [], "competition_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {"sources": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                            for name in SOURCES},
                "generated_script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
                "note": "Attach PV-512 and SAT weights before running; no Kaggle input slug is assumed."}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build())
