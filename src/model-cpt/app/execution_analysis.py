"""
Execution Metrics Tracker
=========================

Collects per-module execution times, API call latencies, and LLM token
usage, then prints a formatted summary table at the end of a pipeline run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class APICallRecord:
    api_name: str
    elapsed_sec: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None


@dataclass
class ModuleMetrics:
    module_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    api_calls: list[APICallRecord] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time


class MetricsTracker:
    def __init__(self) -> None:
        self._modules: dict[str, ModuleMetrics] = {}
        self._module_order: list[str] = []
        self.pipeline_start: float = 0.0
        self.pipeline_end: float = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._modules.clear()
        self._module_order.clear()
        self.pipeline_start = 0.0
        self.pipeline_end = 0.0

    # ------------------------------------------------------------------
    def start_module(self, name: str) -> None:
        if name not in self._modules:
            self._modules[name] = ModuleMetrics(module_name=name)
            self._module_order.append(name)
        self._modules[name].start_time = time.perf_counter()

    def end_module(self, name: str) -> None:
        if name in self._modules:
            self._modules[name].end_time = time.perf_counter()

    # ------------------------------------------------------------------
    def record_api_call(
        self,
        module: str,
        api_name: str,
        elapsed: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        if module not in self._modules:
            self._modules[module] = ModuleMetrics(module_name=module)
            self._module_order.append(module)
        self._modules[module].api_calls.append(
            APICallRecord(
                api_name, elapsed, input_tokens, output_tokens, total_tokens, error
            )
        )

    # ------------------------------------------------------------------
    def print_report(self) -> None:
        """Print a pretty execution-time & token-usage report."""

        SEP = "=" * 78
        THIN = "-" * 78

        print(f"\n{SEP}")
        print("  EXECUTION METRICS REPORT")
        print(SEP)

        grand_input_tokens = 0
        grand_output_tokens = 0
        grand_total_tokens = 0

        for name in self._module_order:
            mod = self._modules[name]
            print(f"\n  Module: {mod.module_name}")
            print(f"  Total execution time: {mod.elapsed:.4f} s")

            if mod.api_calls:
                print(
                    f"  {'API Call':<30} {'Time (s)':>10} {'Status':<12} {'In Tok':>8} {'Out Tok':>8} {'Tot Tok':>8}"
                )
                print(f"  {THIN[:78]}")
                for call in mod.api_calls:
                    status = f"ERROR: {call.error[:30]}" if call.error else "OK"
                    in_t = (
                        str(call.input_tokens) if call.input_tokens is not None else "-"
                    )
                    out_t = (
                        str(call.output_tokens)
                        if call.output_tokens is not None
                        else "-"
                    )
                    tot_t = (
                        str(call.total_tokens) if call.total_tokens is not None else "-"
                    )
                    print(
                        f"  {call.api_name:<30} {call.elapsed_sec:>10.4f} "
                        f"{status:<12} {in_t:>8} {out_t:>8} {tot_t:>8}"
                    )
                    # Show thinking/external tokens if total > input + output
                    if (
                        call.input_tokens is not None
                        and call.output_tokens is not None
                        and call.total_tokens is not None
                    ):
                        io_sum = call.input_tokens + call.output_tokens
                        if call.total_tokens > io_sum:
                            thinking = call.total_tokens - io_sum
                            print(
                                f"  {'':30} {'':>10} {'':12} (incl. {thinking} thinking tokens)"
                            )
                    if call.error:
                        print(f"  {'':30}   !! {call.error}")

                    if call.input_tokens is not None:
                        grand_input_tokens += call.input_tokens
                    if call.output_tokens is not None:
                        grand_output_tokens += call.output_tokens
                    if call.total_tokens is not None:
                        grand_total_tokens += call.total_tokens
            else:
                print("  (no API calls recorded)")

            print(f"  {THIN[:78]}")

        # Final summary
        total_time = self.pipeline_end - self.pipeline_start

        print(f"\n{SEP}")
        print("  FINAL SUMMARY")
        print(SEP)
        print(f"  {'Total pipeline execution time:':<40} {total_time:.4f} s")
        print(f"  {'Total LLM input tokens:':<40} {grand_input_tokens}")
        print(f"  {'Total LLM output tokens:':<40} {grand_output_tokens}")
        print(f"  {'Total LLM tokens:':<40} {grand_total_tokens}")
        io_sum = grand_input_tokens + grand_output_tokens
        if grand_total_tokens > io_sum:
            thinking = grand_total_tokens - io_sum
            print(f"  {'  (incl. thinking tokens):':<40} {thinking}")
        print(SEP + "\n")


# Singleton used throughout the application
tracker = MetricsTracker()
