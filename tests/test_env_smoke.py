from __future__ import annotations

import pytest


@pytest.mark.slow
def test_metadrive_import_available() -> None:
    pytest.importorskip("metadrive")
