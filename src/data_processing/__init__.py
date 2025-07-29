"""
Data processing components for oaSentinel
"""

from .processor import DataProcessor
from .crowdhuman import CrowdHumanProcessor

__all__ = ["DataProcessor", "CrowdHumanProcessor"]