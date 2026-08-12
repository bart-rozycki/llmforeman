"""Make the local test-support module importable under importlib mode."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
