"""Run GPU jobs strictly one at a time (the 8 GB laptop GPU cannot share).

Jobs are lines in results/gpu_jobs.txt: "<name> | <command>".  Finished job names are
recorded in results/gpu_jobs.done, so the queue resumes after a crash, and jobs appended
to the file while it runs are picked up.  A failed job is retried once.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QNAME = sys.argv[1] if len(sys.argv) > 1 else "gpu"
JOBS = ROOT / "results" / f"{QNAME}_jobs.txt"
DONE = ROOT / "results" / f"{QNAME}_jobs.done"
LOGS = ROOT / "results" / "logs" / QNAME


def pending():
    done = {x.replace("__FAILED", "") for x in DONE.read_text().split("\n")} if DONE.exists() else set()
    for line in JOBS.read_text().splitlines():
        if "|" not in line or line.strip().startswith("#"):
            continue
        name, cmd = (x.strip() for x in line.split("|", 1))
        if name not in done:
            return name, cmd
    return None


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    idle = 0
    while True:
        job = pending()
        if job is None:
            idle += 1
            if idle > 120:  # ~1 h without new jobs
                return
            time.sleep(30)
            continue
        idle = 0
        name, cmd = job
        env = dict(__import__("os").environ)
        if QNAME == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        for attempt in (1, 2):
            with open(LOGS / f"{name}.log", "a") as log:
                log.write(f"\n=== {time.ctime()} attempt {attempt}: {cmd}\n")
                log.flush()
                rc = subprocess.call(cmd, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            if rc == 0:
                break
            time.sleep(20)
        with open(DONE, "a") as f:
            f.write(name + ("\n" if rc == 0 else "__FAILED\n"))
        print(f"{time.ctime()} {name} rc={rc}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
