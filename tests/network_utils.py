"""Small unittest helpers for tests that intentionally require external data."""

import os
import unittest


def network_test(test_item):
    """Skip NASA POWER integration tests unless explicitly enabled."""

    return unittest.skipUnless(
        os.environ.get("RUN_NETWORK_TESTS") == "1",
        "NASA POWER network integration test; enable explicitly",
    )(test_item)
