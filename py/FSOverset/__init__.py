from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("FSOverset")
except PackageNotFoundError:
    __version__ = "2026.09"
