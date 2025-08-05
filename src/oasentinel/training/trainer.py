"""
Professional Model Trainer - Strict Implementation
NO fallbacks, NO dummy data, FAIL HARD on errors
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO


class ModelTrainer:
    """
    Professional YOLO model trainer with strict error handling
    """
    
    def __init__(self, model_path: Path, dataset_config: Path):
        """
        Initialize trainer with strict validation
        
        Args:
            model_path: Path to pretrained model (.pt file)
            dataset_config: Path to dataset YAML configuration
        """
        self.model_path = Path(model_path)
        self.dataset_config = Path(dataset_config)
        
        # Strict validation - FAIL if requirements not met
        self._validate_gpu_requirements()
        self._validate_model_file()
        self._validate_dataset_config()
        
        # Load model
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            print(f"FATAL ERROR: Cannot load model {self.model_path}: {e}")
            sys.exit(1)
    
    def _validate_gpu_requirements(self):
        """Validate GPU requirements - FAIL if not met"""
        
        if not torch.cuda.is_available():
            print("FATAL ERROR: CUDA not available. GPU training required.")
            sys.exit(1)
        
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("FATAL ERROR: No GPUs detected.")
            sys.exit(1)
        
        print(f"✓ GPU validation passed: {gpu_count} GPU(s) available")
    
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
    
    def train(self, epochs: int, device: Any, imgsz: int = 640, **kwargs) -> Dict[str, Any]:
        """
        Train model with strict validation
        
        Args:
            epochs: Number of training epochs (REQUIRED)
            device: Training device (REQUIRED)
            imgsz: Image size for training
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        if epochs <= 0:
            print(f"FATAL ERROR: Invalid epochs count: {epochs}")
            sys.exit(1)
        
        print(f"🚀 Starting professional training:")
        print(f"   Model: {self.model_path}")
        print(f"   Dataset: {self.dataset_config}")
        print(f"   Epochs: {epochs}")
        print(f"   Device: {device}")
        print(f"   Image size: {imgsz}")
        
        try:
            results = self.model.train(
                data=str(self.dataset_config),
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                verbose=True,
                **kwargs
            )
            
            print("✅ Training completed successfully!")
            return results
            
        except Exception as e:
            print(f"FATAL ERROR during training: {e}")
            sys.exit(1)
