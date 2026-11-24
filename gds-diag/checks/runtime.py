# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime observation for `runtime-check`.

The other subcommands answer whether a host *can* use GDS. This module answers
whether one run of a workload *did*: it brackets the run with counters and
reports which data path the reads took.

Evidence, from cheapest to most specific:

  nvidia-fs stats   /proc/driver/nvidia-fs/stats "Reads n=" delta. Counts
                    reads served by the nvidia-fs (nvfs) route. Needs
                    rw_stats_enabled=1. Reads 0 on the PCI P2PDMA route, which
                    bypasses nvidia-fs.
  block device      /sys/class/block/<dev>/stat sectors-read delta for the
                    device backing the path. System-wide for that device.
  kernel functions  optional (--kernel-trace, root): ftrace function-profile
                    call counts over the run. pci_p2pdma_map_segment marks a
                    scatter-gather segment mapped to P2P (GPU BAR) memory, the
                    direct path on the P2PDMA route; nvfs_get_p2p_dma_mapping
                    marks the same on the nvidia-fs route; filemap_read marks
                    buffered reads through the page cache. System-wide.

The diagnosis is a pure function of the observation and the platform's active
routes (from `gdscheck -p`), so it is testable without GPU hardware.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .result import CheckResult, GDSMode, Status

NVFS_STATS = "/proc/driver/nvidia-fs/stats"
NVFS_RW_STATS = "/sys/module/nvidia_fs/parameters/rw_stats_enabled"
TRACEFS_CANDIDATES = ("/sys/kernel/tracing", "/sys/kernel/debug/tracing")

P2PDMA_FN = "pci_p2pdma_map_segment"
NVFS_P2P_FN = "nvfs_get_p2p_dma_mapping"
PAGE_CACHE_FN = "filemap_read"
DIRECT_IO_FN = "iomap_dio_rw"
TRACE_FUNCTIONS = (P2PDMA_FN, NVFS_P2P_FN, PAGE_CACHE_FN, DIRECT_IO_FN)

SECTION_WORKLOAD = "Workload"
SECTION_OBSERVED = "Observed Data Path"
SECTION_DIAGNOSIS = "Diagnosis"


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

def parse_nvfs_reads(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (reads, read MiB) from nvidia-fs stats text, or (None, None)."""
    match = re.search(r"^\s*Reads\s*:\s*n=(\d+)\b.*?\breadMiB=(\d+)", text, re.M)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def read_nvfs_reads(path: str = NVFS_STATS) -> Tuple[Optional[int], Optional[int]]:
    try:
        with open(path) as fh:
            return parse_nvfs_reads(fh.read())
    except OSError:
        return None, None


def nvfs_rw_stats_enabled(path: str = NVFS_RW_STATS) -> Optional[bool]:
    """True/False from the nvidia_fs module parameter, None if not loaded."""
    try:
        with open(path) as fh:
            return fh.read().strip() == "1"
    except OSError:
        return None


def parse_block_sectors_read(text: str) -> Optional[int]:
    """Sectors read (third field) from a /sys/class/block/<dev>/stat line."""
    fields = text.split()
    if len(fields) < 3:
        return None
    try:
        return int(fields[2])
    except ValueError:
        return None


def read_block_sectors(device: Optional[str]) -> Optional[int]:
    if not device:
        return None
    name = os.path.basename(device)
    try:
        with open(f"/sys/class/block/{name}/stat") as fh:
            return parse_block_sectors_read(fh.read())
    except OSError:
        return None


def parse_function_profile(texts: List[str]) -> Dict[str, int]:
    """Sum per-CPU ftrace function-profile tables (trace_stat/function*)."""
    counts: Dict[str, int] = {}
    for text in texts:
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 2 or not fields[1].isdigit():
                continue
            counts[fields[0]] = counts.get(fields[0], 0) + int(fields[1])
    return counts


class FunctionProfiler:
    """Count calls to a few kernel functions over one run with the ftrace
    function profiler. Requires root and tracefs. Saves and restores the
    function filter and the profiler switch."""

    def __init__(self, functions: Tuple[str, ...] = TRACE_FUNCTIONS) -> None:
        self.functions = functions
        self.tracefs = next((p for p in TRACEFS_CANDIDATES
                             if os.path.isfile(os.path.join(p, "function_profile_enabled"))), None)
        self.traced: List[str] = []
        self._saved_filter = ""
        self._saved_enabled = "0"

    def unavailable_reason(self) -> Optional[str]:
        if os.geteuid() != 0:
            return "kernel tracing needs root (run gds-diag with sudo -E)"
        if self.tracefs is None:
            return "tracefs with the ftrace function profiler is not available"
        return None

    def _path(self, name: str) -> str:
        return os.path.join(self.tracefs or "", name)

    def _write(self, name: str, value: str) -> None:
        with open(self._path(name), "w") as fh:
            fh.write(value)

    def start(self) -> None:
        with open(self._path("available_filter_functions")) as fh:
            available = {line.split()[0] for line in fh if line.strip()}
        self.traced = [fn for fn in self.functions if fn in available]
        with open(self._path("set_ftrace_filter")) as fh:
            self._saved_filter = fh.read()
        with open(self._path("function_profile_enabled")) as fh:
            self._saved_enabled = fh.read().strip() or "0"
        self._write("function_profile_enabled", "0")
        self._write("set_ftrace_filter", "\n".join(self.traced) + "\n")
        self._write("function_profile_enabled", "1")  # enabling resets the counters

    def stop(self) -> Dict[str, int]:
        try:
            self._write("function_profile_enabled", "0")
            stat_dir = self._path("trace_stat")
            texts = []
            for name in sorted(os.listdir(stat_dir)):
                if name.startswith("function"):
                    with open(os.path.join(stat_dir, name)) as fh:
                        texts.append(fh.read())
            counts = parse_function_profile(texts)
            return {fn: counts.get(fn, 0) for fn in self.traced}
        finally:
            self._restore()

    def _restore(self) -> None:
        try:
            saved = [line for line in self._saved_filter.splitlines()
                     if line.strip() and not line.startswith("#")]
            self._write("set_ftrace_filter", "\n".join(saved) + "\n" if saved else "")
            self._write("function_profile_enabled", self._saved_enabled)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Observation and diagnosis
# ---------------------------------------------------------------------------

@dataclass
class PlatformRoutes:
    """Direct routes `gdscheck -p` reports as active for the path's filesystem."""
    native: Optional[bool] = None    # nvidia-fs (nvfs) route
    p2pdma: Optional[bool] = None    # PCI P2PDMA (or C2C) route
    detail: str = ""


@dataclass
class RuntimeObservation:
    command: str
    exit_code: int
    seconds: float
    device: Optional[str] = None
    device_read_mib: Optional[float] = None
    nvfs_stats_enabled: Optional[bool] = None
    nvfs_reads: Optional[int] = None         # delta over the run
    nvfs_read_mib: Optional[int] = None      # delta over the run
    kernel_counts: Optional[Dict[str, int]] = None
    kernel_trace_note: Optional[str] = None  # why kernel_counts is None


def _routes_text(routes: PlatformRoutes) -> str:
    active = []
    if routes.native:
        active.append("nvidia-fs")
    if routes.p2pdma:
        active.append("PCI P2PDMA")
    return " and ".join(active)


def diagnose(obs: RuntimeObservation, routes: PlatformRoutes) -> Dict[str, List[CheckResult]]:
    """Turn one run's observation into sections of CheckResults."""
    sections: Dict[str, List[CheckResult]] = {
        SECTION_WORKLOAD: [], SECTION_OBSERVED: [], SECTION_DIAGNOSIS: []}

    # --- workload -----------------------------------------------------------
    if obs.exit_code == 0:
        sections[SECTION_WORKLOAD].append(CheckResult(
            check="Workload exit status", mode=GDSMode.COMPAT, status=Status.PASS,
            why=f"The workload exited 0 after {obs.seconds:.1f} s.",
            evidence=obs.command))
    else:
        sections[SECTION_WORKLOAD].append(CheckResult(
            check="Workload exit status", mode=GDSMode.COMPAT, status=Status.FAIL,
            why=f"The workload exited {obs.exit_code}, so its I/O may be incomplete "
                "and the observation below may not represent a full run.",
            mitigation="Fix the workload first; run it once without runtime-check.",
            evidence=obs.command))

    # --- observed data path ---------------------------------------------------
    kc = obs.kernel_counts
    p2pdma_maps = kc.get(P2PDMA_FN) if kc is not None else None
    nvfs_maps = kc.get(NVFS_P2P_FN) if kc is not None else None
    page_cache = kc.get(PAGE_CACHE_FN) if kc is not None else None
    direct_io = kc.get(DIRECT_IO_FN) if kc is not None else None

    if obs.nvfs_reads is not None:
        sections[SECTION_OBSERVED].append(CheckResult(
            check="nvidia-fs reads", mode=GDSMode.NATIVE,
            status=Status.PASS if obs.nvfs_reads > 0 else Status.INFO,
            why=(f"{obs.nvfs_reads} reads ({obs.nvfs_read_mib} MiB) were served by nvidia-fs "
                 "during the run." if obs.nvfs_reads > 0 else
                 "No reads were served by nvidia-fs during the run."),
            evidence=f"{NVFS_STATS}: Reads n= delta {obs.nvfs_reads}"))
    elif obs.nvfs_stats_enabled is False:
        sections[SECTION_OBSERVED].append(CheckResult(
            check="nvidia-fs reads", mode=GDSMode.NATIVE, status=Status.INFO,
            why="nvidia-fs read statistics are disabled, so reads on the nvidia-fs route "
                "were not counted.",
            mitigation=f"Enable them for the next run:\n  echo 1 | sudo tee {NVFS_RW_STATS}"))

    if kc is not None:
        if P2PDMA_FN in kc:
            sections[SECTION_OBSERVED].append(CheckResult(
                check="PCI P2PDMA mappings", mode=GDSMode.P2PDMA,
                status=Status.PASS if p2pdma_maps else Status.INFO,
                why=(f"{p2pdma_maps} scatter-gather segments were mapped to GPU memory by "
                     "the kernel P2PDMA path." if p2pdma_maps else
                     "No segment was mapped to GPU memory by the kernel P2PDMA path."),
                evidence=f"{P2PDMA_FN}: {p2pdma_maps} calls (system-wide)"))
        if NVFS_P2P_FN in kc:
            sections[SECTION_OBSERVED].append(CheckResult(
                check="nvidia-fs P2P mappings", mode=GDSMode.NATIVE,
                status=Status.PASS if nvfs_maps else Status.INFO,
                why=(f"nvidia-fs mapped {nvfs_maps} GPU buffers for P2P DMA." if nvfs_maps else
                     "nvidia-fs mapped no GPU buffer for P2P DMA."),
                evidence=f"{NVFS_P2P_FN}: {nvfs_maps} calls (system-wide)"))
        if page_cache is not None or direct_io is not None:
            sections[SECTION_OBSERVED].append(CheckResult(
                check="Host read path", mode=GDSMode.COMPAT, status=Status.INFO,
                why=f"Buffered reads through the page cache: {page_cache or 0} calls; "
                    f"direct-I/O calls: {direct_io or 0}. Both are system-wide counts.",
                evidence=f"{PAGE_CACHE_FN}={page_cache}, {DIRECT_IO_FN}={direct_io}"))
    elif obs.kernel_trace_note:
        sections[SECTION_OBSERVED].append(CheckResult(
            check="Kernel data-path trace", mode=GDSMode.P2PDMA, status=Status.INFO,
            why=f"Not collected: {obs.kernel_trace_note}. Without it, reads on the PCI "
                "P2PDMA route cannot be seen, because they bypass nvidia-fs.",
            mitigation="Rerun with --kernel-trace as root (sudo -E keeps the environment)."))

    if obs.device_read_mib is not None:
        sections[SECTION_OBSERVED].append(CheckResult(
            check="Device reads", mode=GDSMode.COMPAT, status=Status.INFO,
            why=f"{obs.device_read_mib:.1f} MiB were read from {obs.device} during the run "
                "(all processes using the device).",
            evidence=f"/sys/class/block/{obs.device}/stat sectors-read delta"))

    # --- diagnosis ------------------------------------------------------------
    direct = bool(obs.nvfs_reads) or bool(p2pdma_maps) or bool(nvfs_maps)
    device_read = (obs.device_read_mib or 0) > 0
    route_text = _routes_text(routes)
    p2pdma_unseen = routes.p2pdma and kc is None
    no_evidence = obs.nvfs_reads is None and kc is None

    if direct:
        paths = []
        if obs.nvfs_reads or nvfs_maps:
            paths.append("nvidia-fs")
        if p2pdma_maps:
            paths.append("PCI P2PDMA")
        sections[SECTION_DIAGNOSIS].append(CheckResult(
            check="Direct GDS in this run", mode=GDSMode.P2PDMA if p2pdma_maps else GDSMode.NATIVE,
            status=Status.PASS,
            why=f"Reads reached GPU memory directly through {' and '.join(paths)}. Some reads "
                "may still have taken the host path; this check counts the run as a whole.",
            mitigation=None))
    elif no_evidence or p2pdma_unseen:
        sections[SECTION_DIAGNOSIS].append(CheckResult(
            check="Direct GDS in this run", mode=GDSMode.COMPAT, status=Status.INFO,
            why="There is not enough evidence to tell whether reads went directly to GPU "
                "memory." + (" The PCI P2PDMA route is active and it bypasses the "
                             "nvidia-fs counters." if p2pdma_unseen else ""),
            mitigation="Rerun with --kernel-trace as root, and enable nvidia-fs read "
                       f"statistics:\n  echo 1 | sudo tee {NVFS_RW_STATS}"))
    elif device_read and route_text:
        sections[SECTION_DIAGNOSIS].append(CheckResult(
            check="Host-staged reads despite an active direct route", mode=GDSMode.COMPAT,
            status=Status.WARN,
            why=f"gdscheck reports the {route_text} route active for this filesystem, yet no "
                f"read in this run reached GPU memory directly while {obs.device_read_mib:.1f} "
                "MiB were read from the device. The data went through host memory, so the "
                "cause is in the application or an I/O library rather than the platform.",
            mitigation="Check whether the application or a library stages reads in host "
                       "memory: an I/O library cache (HDF5's raw-data chunk cache stages "
                       "chunks that fit in it), file descriptors opened without O_DIRECT, "
                       "or reads below a library's GDS size threshold. To attribute each "
                       "read to the operation that issued it, trace the run with a "
                       "per-operation tracer such as GDSight "
                       "(https://github.com/izzet/gdsight)."))
    elif device_read:
        sections[SECTION_DIAGNOSIS].append(CheckResult(
            check="Host-staged reads", mode=GDSMode.COMPAT, status=Status.WARN,
            why="No direct GDS route is active for this filesystem, so reads went through "
                "host memory (compat mode).",
            mitigation="Run `gds-diag mount-check <path>` to find what blocks the direct "
                       "routes on this filesystem."))
    else:
        sections[SECTION_DIAGNOSIS].append(CheckResult(
            check="Direct GDS in this run", mode=GDSMode.COMPAT, status=Status.INFO,
            why="No read reached the device during the run. The data may have been cached "
                "in host memory, or the workload did not read from this device.",
            mitigation="Drop the page cache before the run, or check that --path names the "
                       "filesystem the workload reads."))
    return sections
