#!/bin/bash

# oaSentinel Data Processing Script
# Converts datasets to YOLO format and creates train/val/test splits
# Usage: ./scripts/process_data.sh [--dataset crowdhuman] [--splits 0.8,0.15,0.05]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DATA_DIR="$PROJECT_ROOT/data/raw"
PROCESSED_DATA_DIR="$PROJECT_ROOT/data/processed"
SPLITS_DIR="$PROJECT_ROOT/data/splits"
DATASET="crowdhuman"
SPLITS="0.8,0.15,0.05"  # train,val,test
USE_TEST_DATASET=false
AUTO_DETECT=true  # Enable by default for better automation

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${BLUE}===== $1 =====${NC}"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --splits)
            SPLITS="$2"
            shift 2
            ;;
        --use-test-dataset)
            USE_TEST_DATASET=true
            shift
            ;;
        --auto-detect)
            AUTO_DETECT=true
            shift
            ;;
        -h|--help)
            echo "oaSentinel Data Processing Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dataset NAME       Dataset to process (default: crowdhuman)"
            echo "  --splits RATIOS      Train/val/test split ratios (default: 0.8,0.15,0.05)"
            echo "  --use-test-dataset   Force use of test dataset"
            echo "  --auto-detect        Auto-detect best available dataset"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Process CrowdHuman with default splits"
            echo "  $0 --splits 0.7,0.2,0.1             # Custom split ratios"
            echo "  $0 --dataset custom --splits 0.8,0.2,0.0  # Custom dataset, no test split"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
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

log_header "oaSentinel Data Processing"

# Smart dataset detection
if [ "$AUTO_DETECT" = true ] || [ "$USE_TEST_DATASET" = true ]; then
    log_info "Detecting available datasets..."
    
    if [ -f "scripts/detect_dataset.sh" ]; then
        if DATASET_INFO=$(scripts/detect_dataset.sh --format env 2>/dev/null); then
            eval "$DATASET_INFO"
            
            log_info "Dataset detection results:"
            log_info "  Status: $DATASET_STATUS"
            log_info "  Type: $DATASET_TYPE"
            log_info "  Total images: $DATASET_TOTAL_COUNT"
            
            # Override dataset selection based on detection results
            if [ "$USE_TEST_DATASET" = true ] && [ "$DATASET_TYPE" = "test_dataset" ]; then
                log_info "Using test dataset as requested"
                DATASET="test_dataset"
                DATASET_MODE="test"
            elif [ "$DATASET_STATUS" = "ready" ] && [ "$DATASET_TYPE" = "crowdhuman_full" ]; then
                log_info "Using full CrowdHuman dataset"
                DATASET="crowdhuman"
                DATASET_MODE="full"
            elif [ "$DATASET_STATUS" = "test" ] && [ "$DATASET_TYPE" = "test_dataset" ]; then
                log_info "Full dataset not available, using test dataset"
                DATASET="test_dataset"
                DATASET_MODE="test"
            elif [ "$DATASET_STATUS" = "processed" ]; then
                log_success "Dataset already processed, skipping processing"
                exit 0
            else
                log_warning "No suitable dataset found for processing"
                log_info "Available datasets:"
                log_info "  - Run ./scripts/download_data.sh for full CrowdHuman dataset"
                log_info "  - Run ./scripts/create_test_dataset.sh for test dataset"
                log_info "  - Use --use-test-dataset to force test mode"
                exit 1
            fi
        else
            log_warning "Dataset detection failed, using default settings"
            DATASET_MODE="unknown"
        fi
    else
        log_warning "Dataset detection script not found, using default settings"
        DATASET_MODE="unknown"
    fi
else
    DATASET_MODE="manual"
fi

log_info "Processing configuration:"
log_info "  Dataset: $DATASET"
log_info "  Mode: $DATASET_MODE"
log_info "  Input directory: $RAW_DATA_DIR/$DATASET"
log_info "  Output directory: $PROCESSED_DATA_DIR/$DATASET"
log_info "  Split ratios: $SPLITS"

# Create output directories
mkdir -p "$PROCESSED_DATA_DIR/$DATASET"
mkdir -p "$SPLITS_DIR/$DATASET"

# Process CrowdHuman dataset
process_crowdhuman() {
    local input_dir="$RAW_DATA_DIR/crowdhuman"
    local output_dir="$PROCESSED_DATA_DIR/crowdhuman"
    
    log_info "Processing CrowdHuman dataset..."
    
    # Validate input directory
    if [ ! -d "$input_dir" ]; then
        log_error "CrowdHuman dataset not found at $input_dir"
        log_info "Run ./scripts/download_data.sh first"
        exit 1
    fi
    
    # Required files
    local required_files=(
        "$input_dir/annotation_train.odgt"
        "$input_dir/annotation_val.odgt"
        "$input_dir/Images"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -e "$file" ]; then
            log_error "Required file/directory not found: $file"
            exit 1
        fi
    done
    
    log_info "Converting ODGT annotations to YOLO format using CrowdHumanProcessor..."
    
    # Use our custom CrowdHumanProcessor for conversion
    python3 << 'EOF'
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing.crowdhuman import CrowdHumanProcessor

# Configuration from environment
input_dir = Path(os.environ.get('INPUT_DIR', 'data/raw/crowdhuman'))
output_dir = Path(os.environ.get('OUTPUT_DIR', 'data/processed/crowdhuman'))
splits_str = os.environ.get('SPLITS', '0.8,0.15,0.05')

# Parse splits
splits = tuple(float(x) for x in splits_str.split(','))

print(f"Processing CrowdHuman dataset:")
print(f"  Input: {input_dir}")
print(f"  Output: {output_dir}")
print(f"  Splits: {splits}")

try:
    # Initialize processor
    processor = CrowdHumanProcessor(input_dir, output_dir)
    
    # Process dataset
    results = processor.process(splits)
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    # Print summary
    conversion_stats = results.get('conversion_stats', {})
    for split_name, stats in conversion_stats.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  ✅ Converted: {stats['converted_images']} images")
        print(f"  ⚠️  Skipped: {stats['skipped_images']} images") 
        print(f"  ❌ Errors: {stats['error_images']} images")
        print(f"  📊 Annotations: {stats['total_annotations']}")
    
    # Final statistics
    final_stats = results.get('final_statistics', {})
    total_images = final_stats.get('total_images', 0)
    print(f"\n📈 FINAL SUMMARY:")
    print(f"Total images processed: {total_images}")
    print(f"Dataset YAML: {results.get('dataset_yaml', 'N/A')}")
    
    if results.get('custom_splits_applied'):
        print(f"Custom splits applied: {results['split_ratios']}")
    
except Exception as e:
    print(f"ERROR: Processing failed - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nDataset ready for training! 🚀")
EOF
    
    # Set environment variables for Python script
    export INPUT_DIR="$input_dir"
    export OUTPUT_DIR="$output_dir"
    
    log_success "YOLO format conversion complete"
    
    # Create dataset statistics
    log_info "Generating dataset statistics..."
    
    local train_images=$(find "$output_dir/images/train" -name "*.jpg" 2>/dev/null | wc -l)
    local val_images=$(find "$output_dir/images/val" -name "*.jpg" 2>/dev/null | wc -l)
    local total_images=$((train_images + val_images))
    
    # Create statistics file
    cat > "$output_dir/statistics.yaml" << EOF
# CrowdHuman Dataset Statistics (YOLO Format)
dataset: "CrowdHuman"
format: "YOLO"
classes: ["person"]
num_classes: 1

splits:
  train:
    images: $train_images
    labels: $(find "$output_dir/labels/train" -name "*.txt" 2>/dev/null | wc -l)
  validation:
    images: $val_images
    labels: $(find "$output_dir/labels/val" -name "*.txt" 2>/dev/null | wc -l)
  total: $total_images

processing:
  date: "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  input_format: "ODGT"
  output_format: "YOLO"
  
paths:
  images_train: "images/train"
  images_val: "images/val"
  labels_train: "labels/train"
  labels_val: "labels/val"
  dataset_yaml: "crowdhuman.yaml"
EOF
    
    log_success "Dataset statistics generated"
    log_info "Train images: $train_images"
    log_info "Validation images: $val_images"
    log_info "Total images: $total_images"
}

# Apply custom splits if specified
apply_custom_splits() {
    local dataset_dir="$PROCESSED_DATA_DIR/$DATASET"
    local splits_dir="$SPLITS_DIR/$DATASET"
    
    # Parse split ratios
    IFS=',' read -ra RATIO_ARRAY <<< "$SPLITS"
    local train_ratio=${RATIO_ARRAY[0]}
    local val_ratio=${RATIO_ARRAY[1]}
    local test_ratio=${RATIO_ARRAY[2]:-0}
    
    # Validate ratios
    local total_ratio=$(python3 -c "print($train_ratio + $val_ratio + $test_ratio)")
    if (( $(echo "$total_ratio > 1.01" | bc -l) )) || (( $(echo "$total_ratio < 0.99" | bc -l) )); then
        log_error "Split ratios must sum to 1.0 (got $total_ratio)"
        exit 1
    fi
    
    log_info "Applying custom splits: train=$train_ratio, val=$val_ratio, test=$test_ratio"
    
    # Create custom splits using symbolic links (to save space)
    mkdir -p "$splits_dir"/{train,val,test}/{images,labels}
    
    # Get all image files
    local all_images=($(find "$dataset_dir/images" -name "*.jpg" -type f))
    local total_count=${#all_images[@]}
    
    # Calculate split counts
    local train_count=$(python3 -c "print(int($total_count * $train_ratio))")
    local val_count=$(python3 -c "print(int($total_count * $val_ratio))")
    local test_count=$(python3 -c "print($total_count - $train_count - $val_count)")
    
    log_info "Split distribution: train=$train_count, val=$val_count, test=$test_count"
    
    # Shuffle and split
    python3 << EOF
import random
import os
from pathlib import Path

# Get image list
images = [$(printf '"%s",' "${all_images[@]}" | sed 's/,$//')]
random.shuffle(images)

# Split indices
train_end = $train_count
val_end = train_end + $val_count

splits = {
    'train': images[:train_end],
    'val': images[train_end:val_end],
    'test': images[val_end:] if $test_ratio > 0 else []
}

splits_dir = Path("$splits_dir")

for split_name, image_list in splits.items():
    if not image_list:
        continue
        
    print(f"Creating {split_name} split with {len(image_list)} images...")
    
    for image_path in image_list:
        image_path = Path(image_path)
        image_name = image_path.name
        label_name = image_name.replace('.jpg', '.txt')
        
        # Find corresponding label file
        label_path = None
        for labels_dir in Path("$dataset_dir").glob("labels/*"):
            potential_label = labels_dir / label_name
            if potential_label.exists():
                label_path = potential_label
                break
        
        # Create symbolic links
        image_link = splits_dir / split_name / "images" / image_name
        if not image_link.exists():
            image_link.symlink_to(image_path.absolute())
        
        if label_path and label_path.exists():
            label_link = splits_dir / split_name / "labels" / label_name
            if not label_link.exists():
                label_link.symlink_to(label_path.absolute())

print("Custom splits created successfully!")
EOF
    
    # Create dataset YAML for custom splits
    cat > "$splits_dir/dataset.yaml" << EOF
# Custom Split Dataset Configuration
path: $(realpath "$splits_dir")
train: train/images
val: val/images
$([ "$test_ratio" != "0" ] && echo "test: test/images")

nc: 1
names: ['person']

# Split information
splits:
  train_ratio: $train_ratio
  val_ratio: $val_ratio
  test_ratio: $test_ratio
  total_images: $total_count
EOF
    
    log_success "Custom splits applied successfully"
}

# Main processing logic
case $DATASET in
    "crowdhuman")
        process_crowdhuman
        ;;
    *)
        log_error "Unknown dataset: $DATASET"
        log_info "Supported datasets: crowdhuman"
        exit 1
        ;;
esac

# Apply custom splits if different from default
if [ "$SPLITS" != "0.8,0.15,0.05" ] || [[ "$SPLITS" == *",0.0"* ]]; then
    apply_custom_splits
fi

log_header "Processing Complete"
log_success "Dataset '$DATASET' has been processed and is ready for training"
log_info "Processed data location: $PROCESSED_DATA_DIR/$DATASET"
log_info "Next step: ./scripts/train.sh --dataset $DATASET"