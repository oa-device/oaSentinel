"""
Training configuration management for oaSentinel
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration class for model training
    
    Defines all training parameters, model settings, and optimization options
    for the oaSentinel human detection model.
    """
    
    # Model configuration
    model_name: str = "yolo11n.pt"  # Base model to start from
    model_size: str = "n"  # nano, small, medium, large, xlarge
    input_size: int = 640  # Input image size
    
    # Dataset configuration
    dataset_path: Optional[Path] = None  # Path to dataset YAML
    dataset_name: str = "crowdhuman"
    num_classes: int = 1  # Number of classes (person only)
    class_names: List[str] = field(default_factory=lambda: ["person"])
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    patience: int = 10  # Early stopping patience
    save_period: int = 10  # Save checkpoint every N epochs
    
    # Optimization
    optimizer: str = "auto"  # auto, SGD, Adam, AdamW, etc.
    lr_scheduler: str = "auto"  # Learning rate scheduler
    warmup_epochs: int = 3
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Data augmentation
    augment: bool = True
    mixup: bool = False
    copy_paste: bool = False
    
    # Hardware settings
    device: str = "auto"  # auto, cpu, cuda, mps
    workers: int = 8  # Number of data loading workers
    
    # Output settings
    project_name: str = "oaSentinel"
    experiment_name: str = "baseline"
    save_dir: Optional[Path] = None
    
    # Validation settings
    val_split: float = 0.2
    test_split: float = 0.0
    
    # Advanced settings
    resume: bool = False  # Resume from last checkpoint
    pretrained: bool = True  # Use pretrained weights
    freeze: Optional[int] = None  # Freeze first N layers
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["coreml", "onnx"])
    half_precision: bool = False  # Use FP16
    optimize: bool = True  # Optimize for inference
    
    # Logging and monitoring
    verbose: bool = True
    plots: bool = True
    save_json: bool = True
    save_hybrid: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Convert string paths to Path objects
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)
            
        # Set default save directory if not provided
        if self.save_dir is None:
            self.save_dir = Path("models/runs") / self.experiment_name
            
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
            
        # Validate epochs
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive")
            
        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for YAML export"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def get_yolo_args(self) -> Dict[str, Any]:
        """
        Get training arguments in YOLO format
        
        Returns:
            Dictionary of arguments compatible with YOLO training
        """
        return {
            "model": self.model_name,
            "data": str(self.dataset_path) if self.dataset_path else None,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.input_size,
            "lr0": self.learning_rate,
            "patience": self.patience,
            "save_period": self.save_period,
            "optimizer": self.optimizer,
            "warmup_epochs": self.warmup_epochs,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "augment": self.augment,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "device": self.device,
            "workers": self.workers,
            "project": self.project_name,
            "name": self.experiment_name,
            "resume": self.resume,
            "pretrained": self.pretrained,
            "freeze": self.freeze,
            "verbose": self.verbose,
            "plots": self.plots,
            "save_json": self.save_json,
            "save_hybrid": self.save_hybrid,
            "half": self.half_precision,
            "optimize": self.optimize
        }


@dataclass 
class DatasetConfig:
    """Configuration for dataset processing"""
    
    name: str = "crowdhuman"
    input_path: Path = Path("data/raw/crowdhuman")
    output_path: Path = Path("data/processed/crowdhuman")
    
    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.8
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    
    # Processing options
    image_size: int = 640
    augment_training: bool = True
    
    def __post_init__(self):
        """Validate split ratios"""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
            
        # Convert to Path objects
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)