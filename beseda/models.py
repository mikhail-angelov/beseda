"""Model files downloaded on first use."""

import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

log = logging.getLogger("beseda.models")

MODELS_DIR = Path.home() / ".beseda" / "models"


def cached(url: str) -> Path:
    """Downloads `url` once into MODELS_DIR and returns the local file.

    The file name carries a hash of the full URL, so two language packs whose models share a file name
    (…/ru/model.pt, …/de/model.pt) don't overwrite each other. The download goes to a temporary file that
    is renamed only when complete: an interrupted download never leaves a truncated model behind."""
    path = MODELS_DIR / f"{hashlib.sha256(url.encode()).hexdigest()[:8]}-{Path(url).name}"
    if path.exists():
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    fd, partial = tempfile.mkstemp(dir=MODELS_DIR, prefix=f".{path.name}.", suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, path)
    finally:
        Path(partial).unlink(missing_ok=True)
    return path
