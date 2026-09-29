"""Block until a named job appears in results/<queue>_jobs.done (queue dependency).
    python scripts/wait_for_job.py <queue> <job name>"""
import sys, time
from pathlib import Path
done = Path("results") / f"{sys.argv[1]}_jobs.done"
while not (done.exists() and sys.argv[2] in done.read_text().split()):
    time.sleep(60)
print("ready:", sys.argv[2], flush=True)
