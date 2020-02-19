import os
from pathlib import Path


def root_dir():
    return Path(os.path.abspath(os.path.dirname(__file__))).parent.parent