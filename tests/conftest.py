"""Fixtures compartilhadas: um pipeline completo (dados sintéticos) em diretórios temporários."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def trained_env(tmp_path_factory):
    """Roda src.pipeline de ponta a ponta em pastas temporárias (sem tocar em data/ e models/ reais)."""
    base = tmp_path_factory.mktemp("recsys")
    env = {
        **os.environ,
        "RECSYS_DATA_DIR": str(base / "data"),
        "RECSYS_MODELS_DIR": str(base / "models"),
        "RECSYS_REPORTS_DIR": str(base / "reports"),
        "PYTHONPATH": str(ROOT),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--eval-users", "50"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return {"base": base, "env": env}
