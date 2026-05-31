"""Safe pickle save/load with SHA-256 integrity check (R2)."""
import hashlib
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ALLOW_UNSIGNED_ENV = "EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_ALLOW_UNSIGNED"


class IntegrityError(Exception):
    pass


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar_path(path) -> str:
    return str(path) + ".sha256"


def safe_dump(obj: Any, path, protocol: int = 4) -> str:
    path = str(path)
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(path) or ".") as fh:
        tmp = fh.name
        pickle.dump(obj, fh, protocol=protocol)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    digest = _sha256_file(path)
    with open(_sidecar_path(path), "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    log.info("safe_dump: %s sha256=%s", path, digest[:16] + "...")
    return digest


def safe_load(path, *, allow_unsigned: bool | None = None) -> Any:
    path = str(path)
    if allow_unsigned is None:
        allow_unsigned = os.environ.get(ALLOW_UNSIGNED_ENV, "").lower() in (
            "1", "true", "yes"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    sc = _sidecar_path(path)
    if not os.path.exists(sc):
        if allow_unsigned:
            log.warning("safe_load: NO SIDECAR for %s (allow_unsigned)", path)
            with open(path, "rb") as fh:
                return pickle.load(fh)
        raise IntegrityError(
            f"Sidecar {sc} not found. Refusing to unpickle "
            f"(set {ALLOW_UNSIGNED_ENV}=1 to bypass)."
        )

    with open(sc, "r", encoding="utf-8") as fh:
        expected = fh.read().strip().split()[0]
    actual = _sha256_file(path)
    if expected != actual:
        raise IntegrityError(
            f"SHA-256 mismatch for {path}:\n"
            f"  sidecar:  {expected}\n  computed: {actual}\n"
            "File may be corrupted or tampered."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)