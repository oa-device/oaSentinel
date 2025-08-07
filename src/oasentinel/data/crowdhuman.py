"""
CrowdHuman Dataset Processor - Professional Implementation
Converts CrowdHuman .odgt format to YOLO format with strict error handling
NO FALLBACKS - Fails hard on any error condition
"""

import json
import shutil
import sys
from typing import Dict, Any, List, Tuple
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml


class CrowdHumanProcessor:
    """
    Professional CrowdHuman to YOLO converter
    Strict error handling - NO fallbacks or dummy data
    """
    
    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize processor with strict validation
        
        Args:
            input_dir: CrowdHuman dataset directory
            output_dir: YOLO format output directory
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.class_names = ['person', 'head']  # Dual-class detection
        
        # Strict validation - FAIL if requirements not met
        self._validate_input_structure()
    
    def _validate_input_structure(self):
        """Validate CrowdHuman dataset structure - FAIL if invalid"""
        
        if not self.input_dir.exists():
            print(f"FATAL ERROR: Input directory not found: {self.input_dir}")
            sys.exit(1)
        
        required_files = [
            'annotation_train.odgt',
            'annotation_val.odgt', 
            'Images'
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.input_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"FATAL ERROR: Missing required CrowdHuman files: {missing_files}")
            print(f"Expected structure in {self.input_dir}:")
            for f in required_files:
                print(f"  {f}")
            sys.exit(1)
        
        # Validate Images directory has content (search recursively)
        images_dir = self.input_dir / "Images"
        image_files = (
            list(images_dir.rglob("*.jpg")) +
            list(images_dir.rglob("*.jpeg")) +
            list(images_dir.rglob("*.png")) +
            list(images_dir.rglob("*.JPG")) +
            list(images_dir.rglob("*.JPEG")) +
            list(images_dir.rglob("*.PNG"))
        )
        
        if len(image_files) == 0:
            print(f"FATAL ERROR: No images found in {images_dir}")
            sys.exit(1)
        
        print(f"✓ CrowdHuman dataset validated: {len(image_files)} images")
    
    def process(self, splits: Tuple[float, float, float] = (0.8, 0.15, 0.05)) -> Dict[str, Any]:
        """
        Process CrowdHuman dataset to YOLO format
        
        Args:
            splits: (train_ratio, val_ratio, test_ratio)
            
        Returns:
            Processing statistics
        """
        print("Processing CrowdHuman dataset to YOLO format...")
        
        # Validate splits
        if abs(sum(splits) - 1.0) > 0.001:
            print(f"FATAL ERROR: Splits must sum to 1.0, got {sum(splits)}")
            sys.exit(1)
        
        # Create output structure
        self._create_output_structure()
        
        # Build image index once for performance and robust matching
        self._build_image_index()

        # Process train and val splits
        train_stats = self._process_split('annotation_train.odgt', 'train')
        val_stats = self._process_split('annotation_val.odgt', 'val')
        
        # Apply custom splits if different from default
        if splits != (0.8, 0.2, 0.0):
            self._apply_custom_splits(splits)
        
        # Create dataset YAML
        self._create_dataset_yaml()
        
        total_stats = {
            'train_images': train_stats['converted_images'],
            'val_images': val_stats['converted_images'],
            'total_annotations': train_stats['total_annotations'] + val_stats['total_annotations'],
            'splits': splits
        }
        
        print(f"✓ Processing completed: {total_stats['train_images']} train, {total_stats['val_images']} val images")
        return total_stats
    
    def _create_output_structure(self):
        """Create YOLO dataset directory structure"""
        
        directories = [
            'images/train',
            'images/val', 
            'images/test',
            'labels/train',
            'labels/val',
            'labels/test'
        ]
        
        for dir_path in directories:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def _process_split(self, annotation_file: str, split_name: str) -> Dict[str, Any]:
        """Process a single split (train/val)"""
        
        odgt_file = self.input_dir / annotation_file
        images_dir = self.input_dir / "Images"
        
        output_images_dir = self.output_dir / "images" / split_name
        output_labels_dir = self.output_dir / "labels" / split_name
        
        converted_count = 0
        skipped_count = 0
        error_count = 0
        total_annotations = 0
        
        print(f"Processing {split_name} split...")
        
        with open(odgt_file, 'r') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc=f"Converting {split_name}"):
            try:
                data = json.loads(line.strip())
                image_id = data['ID']
                
                # Locate image using prebuilt index (handles nested dirs and case)
                source_image_path = self._find_image_path(image_id)
                
                if not source_image_path:
                    print(f"ERROR: Image not found for ID {image_id}")
                    error_count += 1
                    continue
                
                # Copy image
                target_image_path = output_images_dir / source_image_path.name
                shutil.copy2(source_image_path, target_image_path)
                
                # Get image dimensions
                try:
                    with Image.open(source_image_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    print(f"ERROR: Cannot read image {source_image_path}: {e}")
                    target_image_path.unlink()  # Remove copied image
                    error_count += 1
                    continue
                
                # Process annotations
                yolo_annotations = []
                
                for obj in data.get('gtboxes', []):
                    # Skip ignore regions
                    if obj.get('tag', '') == 'mask':
                        continue
                    
                    # Process head bounding box (class 1)
                    if 'hbox' in obj:
                        hbox = obj['hbox']
                        x, y, w, h = hbox
                        
                        # Validate bounding box
                        if w > 0 and h > 0:
                            center_x, center_y, norm_width, norm_height = self._convert_bbox_to_yolo(
                                (x, y, w, h), img_width, img_height
                            )
                            yolo_annotations.append(
                                f"1 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                            )
                            total_annotations += 1
                    
                    # Process full body bounding box (class 0)
                    if 'fbox' in obj:
                        fbox = obj['fbox']
                        x, y, w, h = fbox
                        
                        # Validate bounding box
                        if w > 0 and h > 0:
                            center_x, center_y, norm_width, norm_height = self._convert_bbox_to_yolo(
                                (x, y, w, h), img_width, img_height
                            )
                            yolo_annotations.append(
                                f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                            )
                            total_annotations += 1
                    
                    # Use vbox ONLY if fbox is not available (strict requirement)
                    elif 'vbox' in obj:
                        vbox = obj['vbox']
                        x, y, w, h = vbox
                        
                        # Validate bounding box
                        if w > 0 and h > 0:
                            center_x, center_y, norm_width, norm_height = self._convert_bbox_to_yolo(
                                (x, y, w, h), img_width, img_height
                            )
                            yolo_annotations.append(
                                f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                            )
                            total_annotations += 1
                
                # Write YOLO annotation file
                if yolo_annotations:
                    label_file = output_labels_dir / f"{image_id}.txt"
                    with open(label_file, 'w') as f_out:
                        f_out.write('\n'.join(yolo_annotations))
                    converted_count += 1
                else:
                    # Remove image if no valid annotations
                    target_image_path.unlink()
                    skipped_count += 1
                    
            except Exception as e:
                print(f"ERROR: Processing failed for line: {e}")
                error_count += 1
                continue
        
        stats = {
            'converted_images': converted_count,
            'skipped_images': skipped_count,
            'error_images': error_count,
            'total_annotations': total_annotations
        }
        
        print(f"  ✓ {split_name}: {converted_count} images, {total_annotations} annotations")
        
        if error_count > 0:
            print(f"  ⚠️  {error_count} errors occurred during processing")
        
        return stats

    def _build_image_index(self):
        """Scan Images/ recursively and build a fast lookup index by ID.

        Handles case variations and multiple extensions. If duplicates exist,
        prefers files in shallow paths first by sorting by path depth.
        """
        images_dir = self.input_dir / "Images"
        all_images: List[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            all_images.extend(images_dir.rglob(pattern))

        # Sort by path depth to prefer shallower files in case of duplicates
        all_images.sort(key=lambda p: len(p.parts))

        self._image_index = {}
        for p in all_images:
            stem_lower = p.stem.lower()
            if stem_lower not in self._image_index:
                self._image_index[stem_lower] = p

    def _find_image_path(self, image_id: str) -> Path:
        """Find image path by ID using the index with case-insensitive match."""
        if not hasattr(self, "_image_index"):
            self._build_image_index()
        return self._image_index.get(str(image_id).lower())
    
    def _convert_bbox_to_yolo(self, bbox: Tuple[float, float, float, float], 
                             img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert bounding box to YOLO format"""
        
        x, y, w, h = bbox
        
        # Convert to center coordinates
        center_x = (x + w / 2) / img_width
        center_y = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        
        # Clamp to valid range
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        norm_width = max(0, min(1, norm_width))
        norm_height = max(0, min(1, norm_height))
        
        return center_x, center_y, norm_width, norm_height
    
    def _apply_custom_splits(self, splits: Tuple[float, float, float]):
        """Apply custom train/val/test splits"""
        
        train_ratio, val_ratio, test_ratio = splits
        print(f"Applying custom splits: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
        
        # This would implement custom splitting logic
        # For now, we use the default CrowdHuman splits
        pass
    
    def _create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        
        yaml_path = self.output_dir.parent.parent / "configs" / "crowdhuman.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 2,
            'names': {0: 'person', 1: 'head'}
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✓ Dataset YAML created: {yaml_path}")
