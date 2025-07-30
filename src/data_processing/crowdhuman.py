"""
CrowdHumanProcessor - Specialized processor for CrowdHuman dataset
"""

import json
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import random

from .processor import DataProcessor


class CrowdHumanProcessor(DataProcessor):
    """
    Processor for CrowdHuman dataset
    
    Converts CrowdHuman .odgt annotation format to YOLO format
    for human detection training.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize CrowdHuman processor
        
        Args:
            input_dir: Directory containing CrowdHuman dataset
            output_dir: Directory for processed YOLO format output
        """
        super().__init__(input_dir, output_dir)
        self.class_names = ['person']  # Single class for human detection
        self.bbox_type = 'vbox'  # Use visible bounding boxes (best for detection)
        
    def validate_input(self) -> bool:
        """
        Validate CrowdHuman dataset structure
        
        Returns:
            True if dataset structure is valid
        """
        required_files = [
            'annotation_train.odgt',
            'annotation_val.odgt',
            'Images'
        ]
        
        for file_name in required_files:
            file_path = self.input_dir / file_name
            if not file_path.exists():
                print(f"ERROR: Required file/directory not found: {file_path}")
                return False
        
        # Check if Images directory contains actual images
        images_dir = self.input_dir / 'Images'
        image_files = list(images_dir.glob('*.jpg'))
        if len(image_files) < 1000:  # Expect at least 1000 images
            print(f"WARNING: Only found {len(image_files)} images, expected ~24,000")
        
        return True
    
    def process(self, splits: Tuple[float, float, float] = (0.8, 0.15, 0.05)) -> Dict[str, Any]:
        """
        Process CrowdHuman dataset and convert to YOLO format
        
        Args:
            splits: Train/validation/test split ratios
            
        Returns:
            Processing results and statistics
        """
        print("Starting CrowdHuman dataset processing...")
        
        # Validate input
        if not self.validate_input():
            raise ValueError("Input validation failed")
        
        # Create output directory structure
        self.create_directory_structure()
        
        # Process train and validation sets
        results = {
            'processing_date': datetime.now().isoformat(),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'bbox_type_used': self.bbox_type,
            'conversion_stats': {},
            'split_stats': {}
        }
        
        # Process training set
        train_stats = self._convert_split(
            self.input_dir / 'annotation_train.odgt',
            'train'
        )
        results['conversion_stats']['train'] = train_stats
        
        # Process validation set  
        val_stats = self._convert_split(
            self.input_dir / 'annotation_val.odgt',
            'val'
        )
        results['conversion_stats']['val'] = val_stats
        
        # Apply custom splits if different from original
        if splits != (0.8, 0.2, 0.0):  # Default CrowdHuman has no test split
            self._apply_custom_splits(splits)
            results['custom_splits_applied'] = True
            results['split_ratios'] = splits
        
        # Create YOLO dataset configuration
        dataset_yaml = self.create_yolo_dataset_yaml(self.class_names)
        results['dataset_yaml'] = str(dataset_yaml)
        
        # Generate and save statistics
        stats = self.generate_statistics()
        results['final_statistics'] = stats
        self.save_statistics(stats)
        
        print(f"Processing complete! Results saved to: {self.output_dir}")
        return results
    
    def _convert_split(self, odgt_file: Path, split_name: str) -> Dict[str, Any]:
        """
        Convert a single ODGT file to YOLO format
        
        Args:
            odgt_file: Path to .odgt annotation file
            split_name: Name of the split (train/val/test)
            
        Returns:
            Conversion statistics
        """
        print(f"Converting {split_name} split from {odgt_file.name}...")
        
        images_dir = self.output_dir / 'images' / split_name
        labels_dir = self.output_dir / 'labels' / split_name
        source_images_dir = self.input_dir / 'Images'
        
        converted_count = 0
        skipped_count = 0
        error_count = 0
        total_annotations = 0
        
        # Read and process each line in the ODGT file
        with open(odgt_file, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc=f"Processing {split_name}"):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                image_id = data['ID']
                image_file = f"{image_id}.jpg"
                
                # Check if source image exists
                source_image_path = source_images_dir / image_file
                if not source_image_path.exists():
                    skipped_count += 1
                    continue
                
                # Get image dimensions
                try:
                    with Image.open(source_image_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    print(f"Error reading image {image_file}: {e}")
                    error_count += 1
                    continue
                
                # Copy image to output directory
                target_image_path = images_dir / image_file
                shutil.copy2(source_image_path, target_image_path)
                
                # Convert annotations
                yolo_annotations = []
                
                for obj in data.get('gtboxes', []):
                    if obj.get('tag') != 'person':
                        continue
                    
                    # Get the appropriate bounding box
                    bbox = obj.get(self.bbox_type)
                    if bbox is None:
                        # Fallback to other box types
                        bbox = obj.get('fbox') or obj.get('hbox')
                        if bbox is None:
                            continue
                    
                    x, y, w, h = bbox
                    
                    # Skip invalid bounding boxes
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Convert to YOLO format
                    center_x, center_y, norm_width, norm_height = self.convert_bbox_to_yolo(
                        (x, y, w, h), img_width, img_height
                    )
                    
                    # Add to annotations (class 0 for person)
                    yolo_annotations.append(
                        f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                    )
                    total_annotations += 1
                
                # Write YOLO annotation file if we have valid annotations
                if yolo_annotations:
                    label_file = labels_dir / f"{image_id}.txt"
                    with open(label_file, 'w') as f_out:
                        f_out.write('\n'.join(yolo_annotations))
                    converted_count += 1
                else:
                    # Remove image if no valid annotations
                    target_image_path.unlink()
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error processing line: {e}")
                error_count += 1
                continue
        
        stats = {
            'converted_images': converted_count,
            'skipped_images': skipped_count,
            'error_images': error_count,
            'total_annotations': total_annotations,
            'source_file': str(odgt_file)
        }
        
        print(f"{split_name} conversion complete:")
        print(f"  ✅ Converted: {converted_count} images")
        print(f"  ⚠️  Skipped: {skipped_count} images")
        print(f"  ❌ Errors: {error_count} images")
        print(f"  📊 Annotations: {total_annotations}")
        
        return stats
    
    def _apply_custom_splits(self, splits: Tuple[float, float, float]):
        """
        Apply custom train/val/test splits using symbolic links
        
        Args:
            splits: Tuple of (train_ratio, val_ratio, test_ratio)
        """
        train_ratio, val_ratio, test_ratio = splits
        
        print(f"Applying custom splits: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
        
        # Collect all processed images
        all_images = []
        for images_dir in [self.output_dir / 'images' / 'train', self.output_dir / 'images' / 'val']:
            if images_dir.exists():
                all_images.extend(list(images_dir.glob('*.jpg')))
        
        # Shuffle for random splits
        random.shuffle(all_images)
        total_count = len(all_images)
        
        # Calculate split indices
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)
        
        splits_data = {
            'train': all_images[:train_end],
            'val': all_images[train_end:val_end],
            'test': all_images[val_end:] if test_ratio > 0 else []
        }
        
        # Create splits directory
        splits_dir = self.output_dir.parent / 'splits' / 'crowdhuman'
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, image_list in splits_data.items():
            if not image_list:
                continue
                
            split_images_dir = splits_dir / split_name / 'images'
            split_labels_dir = splits_dir / split_name / 'labels'
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Creating {split_name} split with {len(image_list)} images...")
            
            for image_path in image_list:
                image_name = image_path.name
                label_name = image_name.replace('.jpg', '.txt')
                
                # Find corresponding label file
                label_path = None
                for labels_dir in self.output_dir.glob('labels/*'):
                    potential_label = labels_dir / label_name
                    if potential_label.exists():
                        label_path = potential_label
                        break
                
                # Create symbolic links
                image_link = split_images_dir / image_name
                if not image_link.exists():
                    image_link.symlink_to(image_path.absolute())
                
                if label_path and label_path.exists():
                    label_link = split_labels_dir / label_name
                    if not label_link.exists():
                        label_link.symlink_to(label_path.absolute())
        
        # Create dataset YAML for custom splits
        splits_yaml = splits_dir / 'dataset.yaml'
        dataset_config = {
            'path': str(splits_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,
            'names': ['person']
        }
        
        if test_ratio > 0:
            dataset_config['test'] = 'test/images'
        
        import yaml
        with open(splits_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Custom splits created at: {splits_dir}")
    
    def generate_annotation_statistics(self) -> Dict[str, Any]:
        """
        Generate detailed annotation statistics
        
        Returns:
            Dictionary containing annotation statistics
        """
        stats = {
            'bbox_distribution': {},
            'annotations_per_image': [],
            'image_dimensions': []
        }
        
        # Analyze processed annotations
        for split in ['train', 'val', 'test']:
            labels_dir = self.output_dir / 'labels' / split
            if not labels_dir.exists():
                continue
                
            for label_file in labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    annotations = f.readlines()
                    stats['annotations_per_image'].append(len(annotations))
                    
                    # Analyze bounding box sizes
                    for ann in annotations:
                        parts = ann.strip().split()
                        if len(parts) >= 5:
                            width = float(parts[3])
                            height = float(parts[4])
                            area = width * height
                            
                            # Categorize by size
                            if area < 0.01:
                                size_cat = 'small'
                            elif area < 0.05:
                                size_cat = 'medium'
                            else:
                                size_cat = 'large'
                            
                            stats['bbox_distribution'][size_cat] = stats['bbox_distribution'].get(size_cat, 0) + 1
        
        return stats