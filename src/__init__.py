"""
oaSentinel - Custom AI models for OrangeAd human detection
"""

__version__ = "0.1.0"
__author__ = "OrangeAd AI Team"
__email__ = "ai@orangead.co"

from .training import TrainingPipeline
from .evaluation import ModelEvaluator  
from .data_processing import DataProcessor

__all__ = [
    "TrainingPipeline",
    "ModelEvaluator", 
    "DataProcessor",
]