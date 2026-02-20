from __future__ import annotations

import time

from .services.job_runner import job_runner


def run_worker_loop(poll_seconds: float = 2.0) -> None:
    while True:
        worked, job_id = job_runner.run_next_queued_job()
        if worked:
            print(f"[worker] processed job: {job_id}")
            continue
        time.sleep(max(0.2, poll_seconds))


if __name__ == "__main__":
    print("[worker] starting queue worker loop")
    run_worker_loop()
