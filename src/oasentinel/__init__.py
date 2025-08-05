"""
oaSentinel - Professional Human Detection Model Training
High-performance AI model development for OrangeAd device ecosystem
"""

__version__ = "1.0.0"
__author__ = "OrangeAd Engineering"

# Strict error handling - no fallbacks
import sys

def _check_requirements():
    """Check critical requirements on import"""
    try:
        import torch
        import ultralytics
        import yaml
        import PIL
    except ImportError as e:
        print(f"FATAL ERROR: Missing required dependency: {e}")
        print("Install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. GPU training will fail.")

# Run requirement check on import
_check_requirements()

# Clean API exports
from .data.crowdhuman import CrowdHumanProcessor
from .training.trainer import ModelTrainer
from .evaluation.evaluator import ModelEvaluator

__all__ = [
    'CrowdHumanProcessor',
    'ModelTrainer', 
    'ModelEvaluator'
]
