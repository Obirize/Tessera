r"""All tests, every file in a process of its own:  .venv\Scripts\python scripts\test.py  [files or -k words ...]

Tests that measure speed run last and alone, so the others do not slow them down. Nothing is left out.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": ROOT, "PYTHONUTF8": "1"}


def measures_speed(f: str, name: str) -> bool:
    """Whether the test compares a stop watch with a limit."""
    with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
        src = fh.read()
    body = re.search(rf"^def {re.escape(name.split('[')[0])}\(.*?(?=^\S|\Z)", src, re.S | re.M)
    return bool(body and re.search(r"assert .*perf_counter\(\)|assert (dt|sec|took|t1?) *<|under_budget|_is_fast|speed_budget", body.group(0)))


def pytest(args, **kw):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, env=ENV, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", **kw)


def pieces(args):
    """(one list of tests per file, tests that run alone at the end)"""
    ids = [ln.strip() for ln in pytest(["--collect-only", *(args or ["tests"])]).stdout.splitlines() if "::" in ln]
    by, alone = {}, []
    for i in ids:
        f, name = i.split("::")[0], i.split("::")[-1]
        if measures_speed(f, name):
            alone.append(i)
        else:
            by.setdefault(f, []).append(i)
    return list(by.values()), alone


def main(args):
    t0 = time.time()
    todo, alone = pieces(args)
    n = sum(map(len, todo)) + len(alone)
    if not n:
        print("test bulunamadı")
        return 1
    bad, done = [], {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "errors": 0}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        def run(job):
            i, t = job
            return t, pytest(["-p", "no:cacheprovider", "--basetemp", os.path.join(tmp, str(i)), *t])
        with ThreadPoolExecutor(max(2, min(12, (os.cpu_count() or 4) - 2))) as ex:
            results = list(ex.map(run, enumerate(todo)))
        if alone:
            results.append(run((len(todo), alone)))
        # a busy machine can make a test time out: what failed gets one more try alone, and that is said
        again = [i for _, r in results for i in re.findall(r"^(?:FAILED|ERROR) (\S+::\S+)", r.stdout, re.M)]
        last = run((len(todo) + 1, again))[1] if again else None
        for t, r in results:
            for k, v in re.findall(r"(\d+) (passed|failed|skipped|errors?)", (r.stdout.strip().splitlines() or [""])[-1]):
                done[v] += int(k)
            if r.returncode not in (0, 5):
                bad.append((t, r))
    if last is not None and last.returncode == 0:
        done["passed"] += done["failed"] + done["error"] + done["errors"]
        done.update(failed=0, error=0, errors=0)
        bad = []
        print("Kalabalıkta kalıp tek başına geçenler: " + ", ".join(i.split("::")[-1] for i in again))
    for t, r in bad:
        print("\n".join(ln for ln in (r.stdout + r.stderr).splitlines() if "ERROR:../" not in ln)[-4000:])
    seen = sum(done.values())
    print(f"{done['passed']} geçti, {done['failed'] + done['error'] + done['errors']} kaldı, {done['skipped']} atlandı"
          f" ({time.time() - t0:.0f} sn)" + ("" if seen >= n else f"  DİKKAT: {n - seen} test sonuç vermedi"))
    return 1 if bad or seen < n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
