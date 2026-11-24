"""Validate 3MF files by slicing them with the Bambu Studio CLI (if installed).

slice_3mf(path) -> dict(ok, return_code, error, plates=[{id, objects, grams, minutes, changes, warning}])
The CLI writes result.json into the output dir; we read it. Uses the settings embedded in the 3MF.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time

BAMBU_EXE = r"C:\Program Files\Bambu Studio\bambu-studio.exe"


def bambu_available() -> bool:
    return os.path.exists(BAMBU_EXE)


def _auto_timeout(path: str) -> int:
    """300 s, or 150 s per plate for big multi-plate / multi-colour files."""
    try:
        import zipfile
        with zipfile.ZipFile(path) as z:
            n = z.read("Metadata/model_settings.config").decode("utf-8", "ignore").count("<plate>")
        return int(max(300, 150 * n))
    except Exception:
        return 300


def slice_3mf(path: str, timeout: int | None = None, keep_dir: str | None = None, retries: int = 1) -> dict:
    if not bambu_available():
        return {"ok": None, "error": "Bambu Studio bulunamadı"}
    timeout = timeout or _auto_timeout(path)
    work = keep_dir or tempfile.mkdtemp(prefix="tessera_slice_")
    os.makedirs(work, exist_ok=True)
    t = time.time()
    try:
        r = subprocess.run([BAMBU_EXE, "--slice", "0", "--outputdir", work, os.path.abspath(path)],
                           cwd=work, capture_output=True, text=True, errors="replace", timeout=timeout)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        if retries > 0:          # the CLI occasionally hangs; one fresh retry usually works
            return slice_3mf(path, timeout, keep_dir, retries - 1)
        return {"ok": False, "error": "zaman aşımı", "seconds": timeout}
    res_path = os.path.join(work, "result.json")
    out = {"ok": False, "return_code": rc, "seconds": round(time.time() - t, 1), "plates": []}
    if os.path.exists(res_path):
        d = json.load(open(res_path, encoding="utf-8"))
        out["return_code"] = d.get("return_code", rc)
        out["error"] = d.get("error_string", "")
        for p in d.get("sliced_plates", []):
            out["plates"].append({
                "id": p.get("id"),
                "objects": len(p.get("objects", [])),
                "grams": round(sum(f.get("total_used_g", 0.0) for f in p.get("filaments", [])), 1),
                "grams_model": round(sum(f.get("main_used_g", 0.0) for f in p.get("filaments", [])), 1),
                "minutes": round(p.get("total_predication", 0.0) / 60.0, 1),
                "changes": p.get("filament_change_times", 0),
                "warning": p.get("warning_message", ""),
            })
        out["ok"] = out["return_code"] == 0 and len(out["plates"]) > 0
        gcodes = [f for f in os.listdir(work) if f.endswith(".gcode")]
        out["gcode_files"] = len(gcodes)
    if keep_dir is None:
        shutil.rmtree(work, ignore_errors=True)
    return out
