"""Interval resource monitoring for local and cloud pipeline workers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

_stop = False

CSV_COLUMNS = [
    "timestamp",
    "step",
    "interval_seconds",
    "system_cpu_percent",
    "memory_used_mib",
    "memory_total_mib",
    "memory_percent",
    "gpu_utilization_percent",
    "gpu_memory_utilization_percent",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_power_watts",
    "gpu_temperature_c",
    "pipeline_cpu_percent",
    "pipeline_rss_mib",
    "inference_cpu_percent",
    "inference_rss_mib",
    "gimbur_cpu_percent",
    "gimbur_rss_mib",
    "gimbur_server_cpu_percent",
    "gimbur_server_rss_mib",
    "prior_batches",
    "prior_states",
    "prior_states_per_second",
    "prior_average_batch_size",
    "prior_average_queue_wait_ms",
    "prior_average_forward_ms",
    "leaf_batches",
    "leaf_states",
    "leaf_states_per_second",
    "leaf_average_batch_size",
    "leaf_average_queue_wait_ms",
    "leaf_average_forward_ms",
    "prior_queue_pending",
    "leaf_queue_pending",
]


def _stop_handler(_signum: int, _frame: object) -> None:
    global _stop
    _stop = True


def _cpu_ticks() -> tuple[int, int]:
    values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return sum(values), values[3] + values[4]


def _process_ticks() -> dict[int, tuple[int, int, int, str]]:
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            status = (entry / "status").read_text().splitlines()
            rss_kib = next(int(line.split()[1]) for line in status if line.startswith("VmRSS:"))
            result[int(entry.name)] = (
                int(fields[13]) + int(fields[14]),
                rss_kib,
                len(list((entry / "task").iterdir())),
                fields[1].strip("()"),
            )
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            continue
    return result


def _memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0])
    total = values["MemTotal"]
    used = total - values["MemAvailable"]
    return {"usedMiB": used / 1024, "totalMiB": total / 1024, "percent": 100 * used / total}


def _gpu() -> dict[str, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        values = [float(value.strip()) for value in output.splitlines()[0].split(",")]
        return dict(
            zip(
                (
                    "utilizationPercent",
                    "memoryUtilizationPercent",
                    "memoryUsedMiB",
                    "memoryTotalMiB",
                    "powerWatts",
                    "temperatureC",
                ),
                values,
            )
        )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _inference(url: str | None) -> dict | None:
    if not url:
        return None
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/diagnostics", timeout=5) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def build_interval_record(
    *,
    started_at: float,
    ended_at: float,
    cpu_start: tuple[int, int],
    cpu_end: tuple[int, int],
    process_start: dict[int, tuple[int, int, int, str]],
    process_end: dict[int, tuple[int, int, int, str]],
    inference_start: dict | None,
    inference_end: dict | None,
    step: str,
) -> dict:
    total_delta = cpu_end[0] - cpu_start[0]
    cpu_percent = 100 * (1 - (cpu_end[1] - cpu_start[1]) / total_delta) if total_delta else 0.0
    clock_ticks = os.sysconf("SC_CLK_TCK")
    processes = []
    for pid, (ticks, rss_kib, threads, name) in process_end.items():
        if pid not in process_start:
            continue
        cpu = 100 * (ticks - process_start[pid][0]) / clock_ticks / max(ended_at - started_at, 1e-9)
        if cpu >= 0.1 or name in ("python", "gimbur", "Gimbur.Server"):
            processes.append(
                {
                    "pid": pid,
                    "name": name,
                    "cpuPercent": cpu,
                    "rssMiB": rss_kib / 1024,
                    "threads": threads,
                }
            )
    inference = None
    if inference_start and inference_end:
        inference = {}
        for name in ("prior", "leaf"):
            before, after = inference_start.get(name, {}), inference_end.get(name, {})
            inference[name] = {
                "batches": after.get("batches", 0) - before.get("batches", 0),
                "states": after.get("states", 0) - before.get("states", 0),
                "statesPerSecond": (
                    after.get("states", 0) - before.get("states", 0)
                )
                / max(ended_at - started_at, 1e-9),
                "averageBatchSize": after.get("average_batch_size", 0),
                "averageQueueWaitMs": after.get("average_queue_wait_ms", 0),
                "averageForwardMs": after.get("average_forward_ms", 0),
            }
        inference["queues"] = inference_end.get("queues", {})
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "step": step,
        "intervalSeconds": ended_at - started_at,
        "systemCpuPercent": cpu_percent,
        "memory": _memory(),
        "gpu": _gpu(),
        "processes": sorted(processes, key=lambda item: item["cpuPercent"], reverse=True),
        "inference": inference,
    }


def _process_role(process: dict) -> str | None:
    name = process["name"]
    if name == "gimbur":
        return "gimbur"
    if name == "Gimbur.Server":
        return "gimbur_server"
    if name == "python":
        return "inference" if process["rssMiB"] > 500 else "pipeline"
    return None


def flatten_record(record: dict) -> dict[str, object]:
    memory = record["memory"]
    gpu = record["gpu"] or {}
    inference = record["inference"] or {}
    prior = inference.get("prior", {})
    leaf = inference.get("leaf", {})
    queues = inference.get("queues", {})
    roles = {
        role: {"cpuPercent": 0.0, "rssMiB": 0.0}
        for role in ("pipeline", "inference", "gimbur", "gimbur_server")
    }
    for process in record["processes"]:
        role = _process_role(process)
        if role:
            roles[role]["cpuPercent"] += process["cpuPercent"]
            roles[role]["rssMiB"] += process["rssMiB"]
    return {
        "timestamp": record["timestamp"],
        "step": record["step"],
        "interval_seconds": record["intervalSeconds"],
        "system_cpu_percent": record["systemCpuPercent"],
        "memory_used_mib": memory["usedMiB"],
        "memory_total_mib": memory["totalMiB"],
        "memory_percent": memory["percent"],
        "gpu_utilization_percent": gpu.get("utilizationPercent", ""),
        "gpu_memory_utilization_percent": gpu.get("memoryUtilizationPercent", ""),
        "gpu_memory_used_mib": gpu.get("memoryUsedMiB", ""),
        "gpu_memory_total_mib": gpu.get("memoryTotalMiB", ""),
        "gpu_power_watts": gpu.get("powerWatts", ""),
        "gpu_temperature_c": gpu.get("temperatureC", ""),
        "pipeline_cpu_percent": roles["pipeline"]["cpuPercent"],
        "pipeline_rss_mib": roles["pipeline"]["rssMiB"],
        "inference_cpu_percent": roles["inference"]["cpuPercent"],
        "inference_rss_mib": roles["inference"]["rssMiB"],
        "gimbur_cpu_percent": roles["gimbur"]["cpuPercent"],
        "gimbur_rss_mib": roles["gimbur"]["rssMiB"],
        "gimbur_server_cpu_percent": roles["gimbur_server"]["cpuPercent"],
        "gimbur_server_rss_mib": roles["gimbur_server"]["rssMiB"],
        "prior_batches": prior.get("batches", ""),
        "prior_states": prior.get("states", ""),
        "prior_states_per_second": prior.get("statesPerSecond", ""),
        "prior_average_batch_size": prior.get("averageBatchSize", ""),
        "prior_average_queue_wait_ms": prior.get("averageQueueWaitMs", ""),
        "prior_average_forward_ms": prior.get("averageForwardMs", ""),
        "leaf_batches": leaf.get("batches", ""),
        "leaf_states": leaf.get("states", ""),
        "leaf_states_per_second": leaf.get("statesPerSecond", ""),
        "leaf_average_batch_size": leaf.get("averageBatchSize", ""),
        "leaf_average_queue_wait_ms": leaf.get("averageQueueWaitMs", ""),
        "leaf_average_forward_ms": leaf.get("averageForwardMs", ""),
        "prior_queue_pending": queues.get("prior_pending", ""),
        "leaf_queue_pending": queues.get("leaf_pending", ""),
    }


def run(output: Path, interval_seconds: float, step: str, inference_url: str | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    while not _stop:
        started = time.monotonic()
        cpu_start = _cpu_ticks()
        process_start = _process_ticks()
        inference_start = _inference(inference_url)
        deadline = started + interval_seconds
        while not _stop and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))
        ended = time.monotonic()
        record = build_interval_record(
            started_at=started,
            ended_at=ended,
            cpu_start=cpu_start,
            cpu_end=_cpu_ticks(),
            process_start=process_start,
            process_end=_process_ticks(),
            inference_start=inference_start,
            inference_end=_inference(inference_url),
            step=step,
        )
        row = flatten_record(record)
        write_header = not output.exists() or output.stat().st_size == 0
        with output.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Log interval resource utilization as CSV.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=120)
    parser.add_argument("--step", required=True)
    parser.add_argument("--inference-url")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        raise SystemExit("interval-seconds must be positive")
    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)
    run(args.output, args.interval_seconds, args.step, args.inference_url)


if __name__ == "__main__":
    main()
