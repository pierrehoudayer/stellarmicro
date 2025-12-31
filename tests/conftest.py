import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-visual",
        action="store_true",
        default=False,
        help="Run visual tests (generates plots).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "visual: mark test as visual/plot-based")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-visual"):
        return
    skip_visual = pytest.mark.skip(reason="need --run-visual to run")
    for item in items:
        if "visual" in item.keywords:
            item.add_marker(skip_visual)
