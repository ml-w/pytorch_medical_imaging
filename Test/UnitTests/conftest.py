import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="Run slow tests (full training-loop tests)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (runs a full training loop)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip = pytest.mark.skip(reason="slow training-loop test; pass --slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
