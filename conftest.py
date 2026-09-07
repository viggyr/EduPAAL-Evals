"""Pytest bootstrap: make the repo root importable (systems/, scenarios/, metrics/)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# mem0 reads MEM0_TELEMETRY at import time (mem0.memory.telemetry). Disabling
# it here keeps the whole test process consistent: besides the PostHog
# phone-home, enabled telemetry opens a second Qdrant client at the fixed
# path ~/.mem0/migrations_qdrant, whose exclusive folder lock breaks any
# test that holds two Memory instances at once. Tests must not depend on
# import order for this. (Production sets the same default in
# Mem0System.__init__ before its first mem0 import.)
os.environ.setdefault("MEM0_TELEMETRY", "false")
