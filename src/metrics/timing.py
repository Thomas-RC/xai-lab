"""Pomocnicze pomiary czasu — context manager."""

import time
from contextlib import contextmanager


@contextmanager
def stopwatch():
    """with stopwatch() as t: ...; print(t())  # ms"""
    t0 = time.perf_counter()
    elapsed_ms = [0.0]

    def get():
        return elapsed_ms[0]

    try:
        yield get
    finally:
        elapsed_ms[0] = (time.perf_counter() - t0) * 1000
