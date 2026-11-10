"""Preserve support for deeply nested extraction paths on Windows."""
import os
from pathlib import Path


def system_path(path):
    if os.name != "nt":
        return Path(path)
    absolute = os.path.abspath(os.fspath(path))
    slash = chr(92)
    prefix = 2 * slash + "?" + slash
    if absolute.startswith(prefix):
        return Path(absolute)
    if absolute.startswith(2 * slash):
        return Path(prefix + "UNC" + slash + absolute[2:])
    return Path(prefix + absolute)
