from __future__ import annotations

import csv
import signal
import subprocess
import sys
import time
from pathlib import Path

from gimbur_nn.monitor import CSV_COLUMNS, build_interval_record, flatten_record


def test_build_interval_record_calculates_cpu_and_process_usage() -> None:
    record = build_interval_record(
        started_at=10.0,
        ended_at=12.0,
        cpu_start=(1000, 400),
        cpu_end=(1200, 450),
        process_start={7: (100, 1024, 2, "gimbur")},
        process_end={7: (200, 2048, 3, "gimbur")},
        inference_start=None,
        inference_end=None,
        step="simulate",
    )

    assert record["systemCpuPercent"] == 75
    process = record["processes"][0]
    assert process["name"] == "gimbur"
    assert process["rssMiB"] == 2
    assert process["threads"] == 3
    flat = flatten_record(record)
    assert list(flat) == CSV_COLUMNS
    assert flat["timestamp"] == record["timestamp"]
    assert flat["gimbur_cpu_percent"] > 0
    assert flat["gimbur_rss_mib"] == 2


def test_monitor_flushes_partial_interval_on_sigterm(tmp_path: Path) -> None:
    output = tmp_path / "monitor.csv"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gimbur_nn.monitor",
            "--output",
            str(output),
            "--interval-seconds",
            "60",
            "--step",
            "test",
        ]
    )
    time.sleep(0.2)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=5) == 0

    with output.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 1
    assert records[0]["step"] == "test"
    assert 0 < float(records[0]["interval_seconds"]) < 5
    assert list(records[0]) == CSV_COLUMNS
