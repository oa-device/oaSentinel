#!/bin/bash

# oaSentinel Test Dataset Creation Script
# Creates a small synthetic dataset for testing the training pipeline
# Usage: ./scripts/create_test_dataset.sh [--samples NUM]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DATA_DIR="$PROJECT_ROOT/data/test_dataset"
SAMPLES=50  # Small test dataset

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${BLUE}===== $1 =====${NC}"; }

show_usage() {
    echo "oaSentinel Test Dataset Creation Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --samples NUM     Number of synthetic samples to generate (default: $SAMPLES)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "This script creates a small synthetic dataset for testing the training pipeline."
    echo "It generates simple images with rectangular 'person' annotations."
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Validate environment
if [ ! -f "pyproject.toml" ]; then
    log_error "Run this script from the oaSentinel project root"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ] && [ ! -f ".venv/bin/activate" ]; then
    log_error "Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Activate virtual environment if needed
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

log_header "oaSentinel Test Dataset Creation"
log_info "Generating $SAMPLES synthetic samples"
log_info "Output directory: $TEST_DATA_DIR"

# Create directory structure
mkdir -p "$TEST_DATA_DIR"/{images,labels}/{train,val}

# Generate synthetic dataset with Python
log_info "Creating synthetic dataset..."

# Set environment variables and run Python
export SAMPLES="$SAMPLES"
export TEST_DATA_DIR="$TEST_DATA_DIR"

python3 -c "
import os
import sys
import random
import numpy as np
from pathlib import Path
import cv2

def create_synthetic_image_with_person(width=640, height=640):
    # Create a random background
    image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Add some noise and texture
    noise = np.random.normal(0, 20, (height, width, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Add some random colored rectangles as 'persons'
    num_persons = random.randint(1, 4)  # 1-4 persons per image
    annotations = []
    
    for _ in range(num_persons):
        # Random person dimensions (typical person aspect ratios)
        person_width = random.randint(60, 150)
        person_height = random.randint(150, 300)
        
        # Random position (ensure it fits in image)
        x = random.randint(0, max(1, width - person_width))
        y = random.randint(0, max(1, height - person_height))
        
        # Add a colored rectangle to represent a person
        color = (
            random.randint(80, 180),
            random.randint(80, 180), 
            random.randint(80, 180)
        )
        cv2.rectangle(image, (x, y), (x + person_width, y + person_height), color, -1)
        
        # Add some variation to make it more realistic
        cv2.rectangle(image, (x, y), (x + person_width, y + person_height), 
                     (color[0]//2, color[1]//2, color[2]//2), 2)
        
        # Convert to YOLO format (normalized coordinates)
        center_x = (x + person_width / 2) / width
        center_y = (y + person_height / 2) / height
        norm_width = person_width / width
        norm_height = person_height / height
        
        # Class 0 for person
        annotations.append(f'0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}')
    
    return image, annotations

# Main execution
samples = int(os.environ.get('SAMPLES', 50))
test_data_dir = Path(os.environ.get('TEST_DATA_DIR'))

print(f'Creating {samples} synthetic samples...')

# Split samples: 80% train, 20% val
train_samples = int(samples * 0.8)
val_samples = samples - train_samples

print(f'Train samples: {train_samples}')
print(f'Validation samples: {val_samples}')

# Create train set
for i in range(train_samples):
    image, annotations = create_synthetic_image_with_person()
    
    # Save image
    image_name = f'train_{i:04d}.jpg'
    image_path = test_data_dir / 'images' / 'train' / image_name
    cv2.imwrite(str(image_path), image)
    
    # Save annotations
    label_name = f'train_{i:04d}.txt'
    label_path = test_data_dir / 'labels' / 'train' / label_name
    
    with open(label_path, 'w') as f:
        f.write('\n'.join(annotations) + '\n')

# Create validation set
for i in range(val_samples):
    image, annotations = create_synthetic_image_with_person()
    
    # Save image
    image_name = f'val_{i:04d}.jpg'
    image_path = test_data_dir / 'images' / 'val' / image_name
    cv2.imwrite(str(image_path), image)
    
    # Save annotations
    label_name = f'val_{i:04d}.txt'
    label_path = test_data_dir / 'labels' / 'val' / label_name
    
    with open(label_path, 'w') as f:
        f.write('\n'.join(annotations) + '\n')

print(f'✅ Created {train_samples} training samples')
print(f'✅ Created {val_samples} validation samples')
print(f'✅ Total: {samples} synthetic samples')
"

if [ $? -ne 0 ]; then
    log_error "Failed to create synthetic dataset"
    exit 1
fi

# Create dataset YAML configuration
log_info "Creating dataset configuration..."

cat > "$TEST_DATA_DIR/dataset.yaml" << EOF
# oaSentinel Test Dataset Configuration
path: $(realpath "$TEST_DATA_DIR")
train: images/train
val: images/val

nc: 1
names: ['person']

# Dataset information
info:
  name: "oaSentinel Test Dataset"
  description: "Synthetic dataset for testing the training pipeline"
  created: "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  samples: $SAMPLES
  format: "YOLO"
  synthetic: true
EOF

# Create statistics file
log_info "Generating dataset statistics..."

train_images=$(find "$TEST_DATA_DIR/images/train" -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
val_images=$(find "$TEST_DATA_DIR/images/val" -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
train_labels=$(find "$TEST_DATA_DIR/labels/train" -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
val_labels=$(find "$TEST_DATA_DIR/labels/val" -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')

cat > "$TEST_DATA_DIR/statistics.yaml" << EOF
# Test Dataset Statistics
dataset: "oaSentinel Test Dataset"
format: "YOLO"
classes: ["person"]
num_classes: 1
synthetic: true

splits:
  train:
    images: $train_images
    labels: $train_labels
  validation:
    images: $val_images
    labels: $val_labels
  total: $((train_images + val_images))

generation:
  date: "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  samples_requested: $SAMPLES
  samples_generated: $((train_images + val_images))
  
paths:
  images_train: "images/train"
  images_val: "images/val"
  labels_train: "labels/train"
  labels_val: "labels/val"
  dataset_yaml: "dataset.yaml"
EOF

log_header "Test Dataset Creation Complete"
log_success "Synthetic dataset created successfully!"
log_info "Location: $TEST_DATA_DIR"
log_info "Train images: $train_images"
log_info "Validation images: $val_images"
log_info "Dataset config: $TEST_DATA_DIR/dataset.yaml"
echo ""
log_info "You can now test the training pipeline with:"
log_info "./scripts/train.sh --model yolo11n --epochs 5 --config configs/test.yaml"