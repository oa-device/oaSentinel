"""
Professional Model Evaluator - Strict Implementation
NO fallbacks, NO dummy metrics, FAIL HARD on errors
"""

import sys
from pathlib import Path
from typing import Dict, Any
import torch
from ultralytics import YOLO


class ModelEvaluator:
    """
    Professional YOLO model evaluator with strict error handling
    """
    
    def __init__(self, model_path: Path, dataset_config: Path):
        """
        Initialize evaluator with strict validation
        
        Args:
            model_path: Path to trained model (.pt file)
            dataset_config: Path to dataset YAML configuration
        """
        self.model_path = Path(model_path)
        self.dataset_config = Path(dataset_config)
        
        # Strict validation - FAIL if requirements not met
        self._validate_model_file()
        self._validate_dataset_config()
        
        # Load model
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            print(f"FATAL ERROR: Cannot load model {self.model_path}: {e}")
            sys.exit(1)
    
    def _validate_model_file(self):
        """Validate model file - FAIL if invalid"""
        
        if not self.model_path.exists():
            print(f"FATAL ERROR: Model file not found: {self.model_path}")
            sys.exit(1)
        
        if not self.model_path.suffix == '.pt':
            print(f"FATAL ERROR: Invalid model format. Expected .pt, got {self.model_path.suffix}")
            sys.exit(1)
        
        print(f"✓ Model file validated: {self.model_path}")
    
    def _validate_dataset_config(self):
        """Validate dataset configuration - FAIL if invalid"""
        
        if not self.dataset_config.exists():
            print(f"FATAL ERROR: Dataset config not found: {self.dataset_config}")
            sys.exit(1)
        
        print(f"✓ Dataset config validated: {self.dataset_config}")
    
    def evaluate(self, split: str = 'val', **kwargs) -> Dict[str, Any]:
        """
        Evaluate model with strict validation
        
        Args:
            split: Dataset split to evaluate on ('val' or 'test')
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results
        """
        if split not in ['val', 'test']:
            print(f"FATAL ERROR: Invalid split '{split}'. Use 'val' or 'test'")
            sys.exit(1)
        
        print(f"🎯 Starting professional evaluation:")
        print(f"   Model: {self.model_path}")
        print(f"   Dataset: {self.dataset_config}")
        print(f"   Split: {split}")
        
        try:
            results = self.model.val(
                data=str(self.dataset_config),
                split=split,
                verbose=True,
                **kwargs
            )
            
            print("✅ Evaluation completed successfully!")
            return results
            
        except Exception as e:
            print(f"FATAL ERROR during evaluation: {e}")
            sys.exit(1)
