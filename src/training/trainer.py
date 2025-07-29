"""
TrainingPipeline - Core training pipeline for oaSentinel models
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

class TrainingPipeline:
    """
    Comprehensive training pipeline for oaSentinel models
    
    Handles YOLO model training with experiment tracking,
    checkpointing, and integration with oaTracker deployment.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize training pipeline
        
        Args:
            config_path: Path to training configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """Create necessary directories for training"""
        dirs = [
            "models/checkpoints",
            "logs/training", 
            "outputs/visualizations"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def initialize_model(self, 
                        architecture: Optional[str] = None,
                        pretrained: Optional[bool] = None,
                        resume_from: Optional[Union[str, Path]] = None) -> YOLO:
        """
        Initialize YOLO model for training
        
        Args:
            architecture: Model architecture (yolov8n, yolov8s, etc.)
            pretrained: Whether to use pretrained weights
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Initialized YOLO model
        """
        # Use config defaults if not provided
        arch = architecture or self.config['model']['architecture']
        is_pretrained = pretrained if pretrained is not None else self.config['model'].get('pretrained', True)
        
        if resume_from and Path(resume_from).exists():
            print(f"Resuming training from: {resume_from}")
            self.model = YOLO(resume_from)
        else:
            if is_pretrained:
                print(f"Loading pretrained {arch} model")
                self.model = YOLO(f"{arch}.pt")
            else:
                print(f"Loading {arch} architecture only")
                self.model = YOLO(f"{arch}.yaml")
        
        return self.model
    
    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Execute training pipeline
        
        Args:
            **kwargs: Override training parameters
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.initialize_model()
        
        # Prepare training arguments
        train_args = self._prepare_training_args(**kwargs)
        
        print("Starting training with configuration:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        
        # Execute training
        results = self.model.train(**train_args)
        
        print("Training completed successfully!")
        return results
    
    def _prepare_training_args(self, **overrides) -> Dict[str, Any]:
        """
        Prepare training arguments from config and overrides
        
        Args:
            **overrides: Parameter overrides
            
        Returns:
            Complete training arguments dictionary
        """
        config = self.config
        dataset_config = config.get('dataset', {})
        training_config = config.get('training', {})
        
        # Find dataset YAML
        dataset_yaml = self._find_dataset_yaml(dataset_config)
        
        args = {
            'data': dataset_yaml,
            'epochs': training_config.get('epochs', 100),
            'batch': dataset_config.get('batch_size', training_config.get('batch_size', 16)),
            'imgsz': dataset_config.get('image_size', 640),
            'device': training_config.get('device', 'auto'),
            'workers': training_config.get('workers', 4),
            'project': 'models/checkpoints',
            'save_period': 10,
            'patience': training_config.get('patience', 10),
            'lr0': training_config.get('learning_rate', 0.001),
            'optimizer': training_config.get('optimizer', 'AdamW'),
            'cos_lr': training_config.get('scheduler') == 'cosine',
            'cache': True,
            'val': True,
        }
        
        # Add augmentation parameters
        augment_config = training_config.get('augment', {})
        args.update(augment_config)
        
        # Apply overrides
        args.update(overrides)
        
        return args
    
    def _find_dataset_yaml(self, dataset_config: Dict[str, Any]) -> str:
        """Find dataset YAML configuration file"""
        dataset_name = dataset_config.get('name', 'crowdhuman')
        dataset_path = dataset_config.get('path', f'data/processed/{dataset_name}')
        
        candidates = [
            f"{dataset_path}/{dataset_name}.yaml",
            f"{dataset_path}/dataset.yaml",
            f"data/splits/{dataset_name}/dataset.yaml"
        ]
        
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        
        raise FileNotFoundError(f"Dataset YAML not found. Tried: {candidates}")
    
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate trained model
        
        Args:
            **kwargs: Evaluation parameters
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Run train() first.")
        
        return self.model.val(**kwargs)
    
    def export(self, formats: list = None, **kwargs) -> Dict[str, str]:
        """
        Export model to deployment formats
        
        Args:
            formats: List of export formats
            **kwargs: Export parameters
            
        Returns:
            Dictionary mapping formats to export paths
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Run train() first.")
        
        formats = formats or ['onnx', 'coreml']
        export_paths = {}
        
        for fmt in formats:
            try:
                export_path = self.model.export(format=fmt, **kwargs)
                export_paths[fmt] = export_path
                print(f"✅ Exported to {fmt}: {export_path}")
            except Exception as e:
                print(f"❌ Failed to export to {fmt}: {e}")
        
        return export_paths