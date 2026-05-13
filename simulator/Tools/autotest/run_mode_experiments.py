#!/usr/bin/env python3
"""Public entry point for residual-fusion SITL mode experiments."""

from __future__ import print_function

import os
import sys


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_pid_piper_mode_experiments import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
