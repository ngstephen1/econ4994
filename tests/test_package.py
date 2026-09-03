"""Smoke tests for the initial package scaffold."""


def test_package_imports() -> None:
    """The source-layout package should be importable during development."""
    import fair_lending

    assert fair_lending.__name__ == "fair_lending"
