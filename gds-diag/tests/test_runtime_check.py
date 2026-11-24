# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from checks import runtime
from checks.result import Status
from checks.runtime import (PlatformRoutes, RuntimeObservation, SECTION_DIAGNOSIS,
                            SECTION_OBSERVED, SECTION_WORKLOAD)

NVFS_STATS_SAMPLE = """\
GDS Version: 1.17.1.13
NVFS statistics(ver: 4.0)
NVFS Driver(version: 2.28.4)
Mellanox PeerDirect Supported: False
IO stats: Enabled, peer IO stats: Disabled
Logging level: info

Active Shadow-Buffer (MiB): 0
Active Process: 0
Reads				: n=256 ok=256 err=0 readMiB=64 io_state_err=0
Reads				: Bandwidth(MiB/s)=812 Avg-Latency(usec)=1203
Sparse Reads			: n=0 io=0 holes=0 pages=0
Writes				: n=0 ok=0 err=0 writeMiB=0 io_state_err=0 pg-cache=0 pg-cache-fail=0 pg-cache-eio=0
"""

PROFILE_CPU0 = """\
  Function                               Hit    Time            Avg             s^2
  --------                               ---    ----            ---             ---
  pci_p2pdma_map_segment                 200    120.3 us        0.601 us        0.010 us
  filemap_read                            90    80.0 us         0.888 us        0.050 us
"""
PROFILE_CPU1 = """\
  Function                               Hit    Time            Avg             s^2
  --------                               ---    ----            ---             ---
  pci_p2pdma_map_segment                  57    34.1 us         0.598 us        0.008 us
"""


def _obs(**kw) -> RuntimeObservation:
    base = dict(command="./app", exit_code=0, seconds=1.0, device="nvme1n1")
    base.update(kw)
    return RuntimeObservation(**base)


def _diag(sections):
    (result,) = sections[SECTION_DIAGNOSIS]
    return result


class ParserTests(unittest.TestCase):
    def test_parse_nvfs_reads(self):
        self.assertEqual(runtime.parse_nvfs_reads(NVFS_STATS_SAMPLE), (256, 64))

    def test_parse_nvfs_reads_missing(self):
        self.assertEqual(runtime.parse_nvfs_reads("NVFS statistics(ver: 4.0)\n"), (None, None))

    def test_parse_block_sectors_read(self):
        line = "  1234  0  98304  560  10  0  80  4  0  700  564  0  0  0  0  0  0\n"
        self.assertEqual(runtime.parse_block_sectors_read(line), 98304)
        self.assertIsNone(runtime.parse_block_sectors_read(""))

    def test_parse_function_profile_sums_cpus(self):
        counts = runtime.parse_function_profile([PROFILE_CPU0, PROFILE_CPU1])
        self.assertEqual(counts["pci_p2pdma_map_segment"], 257)
        self.assertEqual(counts["filemap_read"], 90)
        self.assertNotIn("Function", counts)


class DiagnosisTests(unittest.TestCase):
    def test_nvidia_fs_reads_are_direct(self):
        sections = runtime.diagnose(
            _obs(nvfs_stats_enabled=True, nvfs_reads=256, nvfs_read_mib=64, device_read_mib=64.0),
            PlatformRoutes(native=True, p2pdma=False))
        self.assertEqual(_diag(sections).status, Status.PASS)
        self.assertIn("nvidia-fs", _diag(sections).why)

    def test_p2pdma_mappings_are_direct(self):
        sections = runtime.diagnose(
            _obs(nvfs_stats_enabled=True, nvfs_reads=0, nvfs_read_mib=0, device_read_mib=64.0,
                 kernel_counts={runtime.P2PDMA_FN: 257, runtime.PAGE_CACHE_FN: 180}),
            PlatformRoutes(native=True, p2pdma=True))
        self.assertEqual(_diag(sections).status, Status.PASS)
        self.assertIn("PCI P2PDMA", _diag(sections).why)

    def test_host_staged_with_active_route_points_at_application(self):
        # HDF5 chunk cache on: cuFile is called, but every read is staged through host memory.
        sections = runtime.diagnose(
            _obs(nvfs_stats_enabled=True, nvfs_reads=0, nvfs_read_mib=0, device_read_mib=64.0),
            PlatformRoutes(native=True, p2pdma=False))
        result = _diag(sections)
        self.assertEqual(result.status, Status.WARN)
        self.assertIn("application or an I/O library", result.why)
        self.assertIn("chunk cache", result.mitigation)

    def test_host_staged_without_route_points_at_platform(self):
        sections = runtime.diagnose(
            _obs(nvfs_stats_enabled=True, nvfs_reads=0, nvfs_read_mib=0, device_read_mib=64.0),
            PlatformRoutes(native=False, p2pdma=False))
        result = _diag(sections)
        self.assertEqual(result.status, Status.WARN)
        self.assertIn("mount-check", result.mitigation)

    def test_p2pdma_route_without_kernel_trace_is_inconclusive(self):
        sections = runtime.diagnose(
            _obs(nvfs_stats_enabled=True, nvfs_reads=0, nvfs_read_mib=0, device_read_mib=64.0,
                 kernel_trace_note="--kernel-trace was not given"),
            PlatformRoutes(native=True, p2pdma=True))
        result = _diag(sections)
        self.assertEqual(result.status, Status.INFO)
        self.assertIn("bypasses the nvidia-fs counters", result.why)

    def test_no_evidence_is_inconclusive(self):
        sections = runtime.diagnose(
            _obs(nvfs_stats_enabled=False, device_read_mib=64.0,
                 kernel_trace_note="--kernel-trace was not given"),
            PlatformRoutes(native=True, p2pdma=False))
        self.assertEqual(_diag(sections).status, Status.INFO)
        checks = [r.check for r in sections[SECTION_OBSERVED]]
        self.assertIn("nvidia-fs reads", checks)       # stats-disabled hint
        self.assertIn("Kernel data-path trace", checks)

    def test_no_device_reads(self):
        sections = runtime.diagnose(
            _obs(nvfs_stats_enabled=True, nvfs_reads=0, nvfs_read_mib=0, device_read_mib=0.0),
            PlatformRoutes(native=True, p2pdma=False))
        self.assertEqual(_diag(sections).status, Status.INFO)
        self.assertIn("did not read", _diag(sections).why)

    def test_failed_workload_is_fail(self):
        sections = runtime.diagnose(_obs(exit_code=2, nvfs_stats_enabled=True, nvfs_reads=0,
                                         nvfs_read_mib=0, device_read_mib=0.0),
                                    PlatformRoutes(native=True))
        (workload,) = sections[SECTION_WORKLOAD]
        self.assertEqual(workload.status, Status.FAIL)


class RenderingTests(unittest.TestCase):
    def test_pass_verdict_and_counts_shown_in_normal_mode(self):
        from subcommands import runtime_check
        sections = runtime.diagnose(
            _obs(nvfs_stats_enabled=True, nvfs_reads=256, nvfs_read_mib=64, device_read_mib=64.0),
            PlatformRoutes(native=True, p2pdma=False))
        text = "\n".join(runtime_check.render_report_sections(sections, verbose=False))
        self.assertIn("Direct GDS in this run", text)
        self.assertIn("256 reads (64 MiB)", text)
        self.assertNotIn("Workload exit status", text)  # a passing workload is not repeated

    def test_failed_workload_is_shown(self):
        from subcommands import runtime_check
        sections = runtime.diagnose(_obs(exit_code=1, nvfs_stats_enabled=True, nvfs_reads=0,
                                         nvfs_read_mib=0, device_read_mib=0.0),
                                    PlatformRoutes(native=True))
        text = "\n".join(runtime_check.render_report_sections(sections, verbose=False))
        self.assertIn("Workload exit status", text)

    def test_tracer_url_renders_on_its_own_line(self):
        from checks.output import _format_block_text
        sections = runtime.diagnose(
            _obs(nvfs_stats_enabled=True, nvfs_reads=0, nvfs_read_mib=0, device_read_mib=64.0),
            PlatformRoutes(native=True, p2pdma=False))
        lines = [line.strip() for line in _format_block_text(_diag(sections).mitigation)]
        self.assertIn("https://github.com/izzet/gdsight", lines)
        self.assertNotIn("(", lines)
        self.assertNotIn(").", lines)


if __name__ == "__main__":
    unittest.main()
