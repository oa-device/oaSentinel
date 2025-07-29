"""
DataProcessor - Base data processing functionality for oaSentinel
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yaml
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    """
    Abstract base class for dataset processors
    
    Provides common functionality for converting various
    annotation formats to YOLO format for training.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize data processor
        
        Args:
            input_dir: Directory containing raw dataset
            output_dir: Directory for processed output
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.statistics = {}
    
    @abstractmethod
    def process(self, splits: Tuple[float, float, float] = (0.8, 0.15, 0.05)) -> Dict[str, Any]:
        """
        Process dataset and convert to YOLO format
        
        Args:
            splits: Train/validation/test split ratios
            
        Returns:
            Processing results and statistics
        """
        pass
    
    @abstractmethod
    def validate_input(self) -> bool:
        """
        Validate input dataset structure and files
        
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    def create_yolo_dataset_yaml(self, 
                                class_names: List[str],
                                train_path: str = "images/train",
                                val_path: str = "images/val", 
                                test_path: Optional[str] = None) -> Path:
        """
        Create YOLO dataset configuration file
        
        Args:
            class_names: List of class names
            train_path: Relative path to training images
            val_path: Relative path to validation images
            test_path: Relative path to test images (optional)
            
        Returns:
            Path to created dataset YAML file
        """
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': train_path,
            'val': val_path,
            'nc': len(class_names),
            'names': class_names
        }
        
        if test_path:
            dataset_config['test'] = test_path
        
        yaml_path = self.output_dir / f"{self.input_dir.name}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return yaml_path
    
    def generate_statistics(self) -> Dict[str, Any]:
        """
        Generate dataset statistics after processing
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'dataset_name': self.input_dir.name,
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'processing_date': None,  # Will be set by subclass
            'format': 'YOLO',
            'splits': {},
            'total_images': 0,
            'total_annotations': 0,
            'classes': []
        }
        
        # Count images and labels in each split
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / 'images' / split
            labels_dir = self.output_dir / 'labels' / split
            
            if images_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg')))
                label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
                
                stats['splits'][split] = {
                    'images': image_count,
                    'labels': label_count
                }
                stats['total_images'] += image_count
        
        return stats
    
    def save_statistics(self, stats: Dict[str, Any]):
        """
        Save processing statistics to file
        
        Args:
            stats: Statistics dictionary to save
        """
        stats_file = self.output_dir / 'statistics.yaml'
        with open(stats_file, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        print(f"Statistics saved to: {stats_file}")
    
    def convert_bbox_to_yolo(self, 
                           bbox: Tuple[float, float, float, float],
                           img_width: int, 
                           img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert bounding box to YOLO format
        
        Args:
            bbox: Bounding box in (x, y, width, height) format
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            YOLO format bounding box (center_x, center_y, width, height) normalized
        """
        x, y, w, h = bbox
        
        # Convert to center coordinates
        center_x = (x + w / 2) / img_width
        center_y = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        
        # Ensure values are within [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        norm_width = max(0, min(1, norm_width))
        norm_height = max(0, min(1, norm_height))
        
        return center_x, center_y, norm_width, norm_height
    
    def create_directory_structure(self):
        """Create output directory structure for YOLO format"""
        dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Created directory structure in: {self.output_dir}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing"""
        # Override in subclasses if needed
        pass