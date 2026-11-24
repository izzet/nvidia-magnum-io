# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
runtime-check subcommand.

Runs a workload and reports which data path its reads actually took: direct to
GPU memory (nvidia-fs or PCI P2PDMA) or through host memory. Combined with the
platform's active routes from `gdscheck -p`, it separates platform causes from
application causes: a host-staged run on a path whose direct route is active
points at the application or an I/O library.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

from ._base import Subcommand

_DESCRIPTION = """\
Run a workload and report the data path its reads took.

Usage:
  gds-diag.py runtime-check --path /mnt/data -- ./my_app --input /mnt/data/x.h5
  sudo -E python3 gds-diag.py runtime-check --kernel-trace --path /mnt/data -- ./my_app

Evidence collected around the run:
  - nvidia-fs read counter delta (/proc/driver/nvidia-fs/stats; needs
    rw_stats_enabled=1). Blind to the PCI P2PDMA route.
  - Reads from the block device backing --path (/sys/class/block/<dev>/stat).
  - With --kernel-trace (root): ftrace call counts of pci_p2pdma_map_segment,
    nvfs_get_p2p_dma_mapping, filemap_read and iomap_dio_rw.

Platform context: the direct routes gdscheck -p reports as active for the
filesystem of --path.

Counters are system-wide for the device and the kernel. Run on a quiet host
and drop the page cache first for a clean reading. When run with sudo, the
workload runs as the invoking user.
"""


def _platform_routes(path: str, fs_type: str):
    from checks import gds_report as gp
    from checks.gdscheck import (_find_gdscheck, _run_gdscheck_raw,
                                 driver_config_has_direct_p2pdma_token,
                                 driver_config_has_native,
                                 driver_config_status_supported)
    from checks.runtime import PlatformRoutes

    gdscheck = _find_gdscheck()
    if not gdscheck:
        return PlatformRoutes(detail="gdscheck not found")
    output, err = _run_gdscheck_raw(gdscheck, required_section="DRIVER CONFIGURATION")
    if output is None:
        return PlatformRoutes(detail=err or "gdscheck -p failed")
    keys = gp._FS_DRIVER_KEYS.get(fs_type, ())
    modes = gp._driver_config_modes_any(output, keys) if keys else None
    if modes is None:
        return PlatformRoutes(detail=f"no DRIVER CONFIGURATION entry for {fs_type}")
    native = driver_config_has_native(modes)
    p2pdma = driver_config_has_direct_p2pdma_token(modes)
    for key in keys:  # older gdscheck prints P2PDMA on its own line, e.g. "NVMe P2PDMA : Supported"
        separate = gp._driver_config_modes(output, f"{key} P2PDMA")
        if separate is not None and driver_config_status_supported(separate):
            p2pdma = True
    return PlatformRoutes(native=native, p2pdma=p2pdma,
                          detail=f"{', '.join(keys)}: {modes}")


def _demote_to_invoking_user():
    """Run the workload as the sudo-invoking user, not root."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if os.geteuid() != 0 or not uid or not gid:
        return None

    def _demote():
        os.setgid(int(gid))
        os.setuid(int(uid))
    return _demote


def _observe(args: argparse.Namespace, command: list, device):
    from checks import runtime

    stats_enabled = runtime.nvfs_rw_stats_enabled()
    nvfs_before = runtime.read_nvfs_reads() if stats_enabled else (None, None)
    sectors_before = runtime.read_block_sectors(device)

    profiler = None
    trace_note = None
    if args.kernel_trace:
        profiler = runtime.FunctionProfiler()
        trace_note = profiler.unavailable_reason()
        if trace_note:
            profiler = None
    else:
        trace_note = "--kernel-trace was not given"

    counts = None
    if profiler:
        profiler.start()
    start = time.monotonic()
    try:
        # With --json, stdout carries only the report: send the workload's stdout to stderr.
        proc = subprocess.run(command, preexec_fn=_demote_to_invoking_user(),
                              stdout=sys.stderr if args.json else None)
        exit_code = proc.returncode
    except OSError as exc:
        print(f"ERROR: could not run {command[0]}: {exc}", file=sys.stderr)
        exit_code = 127
    finally:
        seconds = time.monotonic() - start
        if profiler:
            counts = profiler.stop()

    nvfs_after = runtime.read_nvfs_reads() if stats_enabled else (None, None)
    sectors_after = runtime.read_block_sectors(device)

    def _delta(after, before):
        return after - before if after is not None and before is not None else None

    sectors = _delta(sectors_after, sectors_before)
    return runtime.RuntimeObservation(
        command=" ".join(shlex.quote(c) for c in command),
        exit_code=exit_code,
        seconds=seconds,
        device=os.path.basename(device) if device else None,
        device_read_mib=sectors * 512 / (1 << 20) if sectors is not None else None,
        nvfs_stats_enabled=stats_enabled,
        nvfs_reads=_delta(nvfs_after[0], nvfs_before[0]),
        nvfs_read_mib=_delta(nvfs_after[1], nvfs_before[1]),
        kernel_counts=counts,
        kernel_trace_note=None if counts is not None else trace_note,
    )


def _run(args: argparse.Namespace) -> int:
    from checks import runtime
    from checks.fs_matrix import get_backing_device, get_fs_type
    from checks.output import (bold, render_mitigation_plan, render_sections,
                               render_summary, results_to_json)
    from checks.version import tool_metadata

    command = list(args.workload)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("ERROR: no workload given. Usage: runtime-check --path PATH -- COMMAND [ARGS...]",
              file=sys.stderr)
        return 3

    path = os.path.realpath(args.path)
    fs_type = get_fs_type(path)
    if not fs_type:
        print(f"ERROR: could not detect the filesystem type for {path}", file=sys.stderr)
        return 3
    device = get_backing_device(path)
    routes = _platform_routes(path, fs_type)

    obs = _observe(args, command, device)
    sections = runtime.diagnose(obs, routes)

    if args.json:
        out = {
            "tool": tool_metadata(),
            "mode": "runtime-check",
            "path": path,
            "fs_type": fs_type,
            "device": obs.device,
            "platform_routes": {"native": routes.native, "p2pdma": routes.p2pdma,
                                "detail": routes.detail},
            "observation": {
                "command": obs.command,
                "exit_code": obs.exit_code,
                "seconds": round(obs.seconds, 3),
                "device_read_mib": obs.device_read_mib,
                "nvfs_stats_enabled": obs.nvfs_stats_enabled,
                "nvfs_reads": obs.nvfs_reads,
                "nvfs_read_mib": obs.nvfs_read_mib,
                "kernel_counts": obs.kernel_counts,
                "kernel_trace_note": obs.kernel_trace_note,
            },
            "sections": {title: results_to_json(results) for title, results in sections.items()},
        }
        print(json.dumps(out, indent=2))
    else:
        print(bold("  GDS Runtime Check"))
        print(f"  Path        : {path} ({fs_type}{', ' + obs.device if obs.device else ''})")
        print(f"  Routes      : {routes.detail}")
        print(f"  Workload    : {obs.command}")
        print()
        print("\n".join(render_sections(sections, verbose=args.verbose)))
        print(render_summary(sections))
        print()
        print("\n".join(render_mitigation_plan(sections)))

    if obs.exit_code != 0:
        return 1
    return 0


class RuntimeCheckCommand(Subcommand):
    name = "runtime-check"
    help = "run a workload and report the data path its reads took"
    description = _DESCRIPTION
    order = 45

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--path", required=True,
            help="a path on the filesystem the workload reads (selects the device and route)",
        )
        parser.add_argument(
            "--kernel-trace", action="store_true",
            help="count kernel data-path functions with ftrace (root); needed to see the "
                 "PCI P2PDMA route",
        )
        parser.add_argument(
            "workload", nargs=argparse.REMAINDER,
            help="the command to run, after --",
        )

    def run(self, args: argparse.Namespace) -> int:
        return _run(args)


COMMAND = RuntimeCheckCommand()
