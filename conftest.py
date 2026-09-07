"""Pytest bootstrap: make the repo root importable (systems/, scenarios/, metrics/)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
