"""
oaSentinel - Professional Human Detection Model Training
High-performance AI model development for OrangeAd device ecosystem
"""

__version__ = "1.0.0"
__author__ = "OrangeAd Engineering"

import sys

def _check_basic_requirements():
    """Check only lightweight, universal requirements.

    Heavy training dependencies (torch/ultralytics) are intentionally NOT
    imported here to avoid side-effects when the package is imported for
    non-training tasks (e.g., dataset processing). Training and evaluation
    entrypoints perform their own strict validation.
    """
    try:
        import yaml  # noqa: F401
        import PIL   # noqa: F401
    except ImportError as e:
        print(f"FATAL ERROR: Missing required dependency: {e}")
        print("Install requirements: pip install -r requirements.txt")
        sys.exit(1)

# Do not run checks on import; CLI tools validate per-task.

# Clean API exports (avoid importing training/eval at package import time)
from .data.crowdhuman import CrowdHumanProcessor

__all__ = [
    'CrowdHumanProcessor'
]
