from __future__ import annotations

import shutil

import pytest


@pytest.fixture(scope="session")
def spark():
    if shutil.which("java") is None:
        pytest.skip("Java runtime required for pyspark")
    from mua.spark import get_spark

    return get_spark("mua-test")
