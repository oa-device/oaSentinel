#!/bin/bash

# oaSentinel Smart Dataset Detection Script
# Detects available datasets and returns configuration information
# Usage: ./scripts/detect_dataset.sh [--verbose] [--format json|yaml|env]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERBOSE=false
OUTPUT_FORMAT="env"  # env, json, yaml

# Logging functions
log_info() { [ "$VERBOSE" = true ] && echo -e "${BLUE}[INFO]${NC} $1" >&2; }
log_success() { [ "$VERBOSE" = true ] && echo -e "${GREEN}[SUCCESS]${NC} $1" >&2; }
log_warning() { [ "$VERBOSE" = true ] && echo -e "${YELLOW}[WARNING]${NC} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

show_usage() {
    echo "oaSentinel Smart Dataset Detection Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --verbose         Enable verbose output"
    echo "  --format FORMAT   Output format: env, json, yaml (default: env)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "This script detects available datasets and returns configuration information."
    echo "It checks for datasets in order of preference and returns the best available option."
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --format)
            OUTPUT_FORMAT="$2"
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

# Dataset detection functions
detect_crowdhuman_full() {
    local dataset_dir="$PROJECT_ROOT/data/raw/crowdhuman"
    local images_dir="$dataset_dir/Images"
    local train_ann="$dataset_dir/annotation_train.odgt"
    local val_ann="$dataset_dir/annotation_val.odgt"
    
    if [ -d "$images_dir" ] && [ -f "$train_ann" ] && [ -f "$val_ann" ]; then
        local image_count=$(find "$images_dir" -name "*.jpg" 2>/dev/null | wc -l)
        local train_count=$(wc -l < "$train_ann" 2>/dev/null || echo "0")
        local val_count=$(wc -l < "$val_ann" 2>/dev/null || echo "0")
        
        # Full dataset should have thousands of images
        if [ "$image_count" -gt 1000 ] && [ "$train_count" -gt 1000 ]; then
            echo "crowdhuman_full:$dataset_dir:$image_count:$train_count:$val_count"
            return 0
        elif [ "$image_count" -gt 0 ]; then
            echo "crowdhuman_partial:$dataset_dir:$image_count:$train_count:$val_count"
            return 0
        fi
    fi
    return 1
}

detect_test_dataset() {
    local test_dir="$PROJECT_ROOT/data/test_dataset"
    local images_train="$test_dir/images/train"
    local images_val="$test_dir/images/val"
    local labels_train="$test_dir/labels/train"
    local labels_val="$test_dir/labels/val"
    local config_file="$test_dir/dataset.yaml"
    
    if [ -d "$images_train" ] && [ -d "$images_val" ] && [ -d "$labels_train" ] && [ -d "$labels_val" ]; then
        local train_count=$(find "$images_train" -name "*.jpg" 2>/dev/null | wc -l)
        local val_count=$(find "$images_val" -name "*.jpg" 2>/dev/null | wc -l)
        local total_count=$((train_count + val_count))
        
        if [ "$total_count" -gt 0 ]; then
            echo "test_dataset:$test_dir:$total_count:$train_count:$val_count"
            return 0
        fi
    fi
    return 1
}

detect_processed_dataset() {
    local processed_dir="$PROJECT_ROOT/data/processed/crowdhuman"
    local splits_dir="$PROJECT_ROOT/data/splits/crowdhuman"
    
    if [ -d "$processed_dir" ] || [ -d "$splits_dir" ]; then
        local image_count=0
        local train_count=0
        local val_count=0
        
        # Count processed images
        if [ -d "$processed_dir/images" ]; then
            image_count=$(find "$processed_dir/images" -name "*.jpg" 2>/dev/null | wc -l)
        fi
        
        # Count split images
        if [ -d "$splits_dir/train/images" ]; then
            train_count=$(find "$splits_dir/train/images" -name "*.jpg" 2>/dev/null | wc -l)
        fi
        
        if [ -d "$splits_dir/val/images" ]; then
            val_count=$(find "$splits_dir/val/images" -name "*.jpg" 2>/dev/null | wc -l)
        fi
        
        local total_count=$((image_count + train_count + val_count))
        if [ "$total_count" -gt 0 ]; then
            echo "processed_dataset:$processed_dir:$total_count:$train_count:$val_count"
            return 0
        fi
    fi
    return 1
}

# Main detection logic
log_info "Detecting available datasets..."

DATASET_INFO=""
DATASET_TYPE=""
DATASET_PATH=""
DATASET_CONFIG=""
DATASET_STATUS="none"

# Try detection in order of preference
if DATASET_INFO=$(detect_crowdhuman_full); then
    log_success "Full CrowdHuman dataset detected"
    DATASET_STATUS="ready"
elif DATASET_INFO=$(detect_processed_dataset); then
    log_success "Processed dataset detected"
    DATASET_STATUS="processed"
elif DATASET_INFO=$(detect_test_dataset); then
    log_success "Test dataset detected"
    DATASET_STATUS="test"
else
    log_warning "No suitable dataset found"
    DATASET_STATUS="none"
fi

# Parse dataset info
if [ -n "$DATASET_INFO" ]; then
    IFS=':' read -r DATASET_TYPE DATASET_PATH TOTAL_COUNT TRAIN_COUNT VAL_COUNT <<< "$DATASET_INFO"
    
    # Determine appropriate config file
    case "$DATASET_TYPE" in
        "crowdhuman_full"|"crowdhuman_partial")
            DATASET_CONFIG="$PROJECT_ROOT/crowdhuman.yaml"
            ;;
        "test_dataset")
            DATASET_CONFIG="$PROJECT_ROOT/data/test_dataset/dataset.yaml"
            ;;
        "processed_dataset")
            # Look for existing config or use crowdhuman.yaml
            DATASET_CONFIG="$PROJECT_ROOT/crowdhuman.yaml"
            ;;
    esac
    
    log_info "Dataset type: $DATASET_TYPE"
    log_info "Dataset path: $DATASET_PATH"
    log_info "Total images: $TOTAL_COUNT"
    log_info "Training images: $TRAIN_COUNT"
    log_info "Validation images: $VAL_COUNT"
    log_info "Config file: $DATASET_CONFIG"
else
    DATASET_TYPE="none"
    DATASET_PATH=""
    DATASET_CONFIG=""
    TOTAL_COUNT=0
    TRAIN_COUNT=0
    VAL_COUNT=0
fi

# Output results in requested format
case "$OUTPUT_FORMAT" in
    "env")
        echo "DATASET_STATUS=\"$DATASET_STATUS\""
        echo "DATASET_TYPE=\"$DATASET_TYPE\""
        echo "DATASET_PATH=\"$DATASET_PATH\""
        echo "DATASET_CONFIG=\"$DATASET_CONFIG\""
        echo "DATASET_TOTAL_COUNT=\"$TOTAL_COUNT\""
        echo "DATASET_TRAIN_COUNT=\"$TRAIN_COUNT\""
        echo "DATASET_VAL_COUNT=\"$VAL_COUNT\""
        ;;
    "json")
        cat << EOF
{
  "status": "$DATASET_STATUS",
  "type": "$DATASET_TYPE",
  "path": "$DATASET_PATH",
  "config": "$DATASET_CONFIG",
  "counts": {
    "total": $TOTAL_COUNT,
    "train": $TRAIN_COUNT,
    "val": $VAL_COUNT
  },
  "ready_for_training": $([ "$DATASET_STATUS" != "none" ] && echo "true" || echo "false")
}
EOF
        ;;
    "yaml")
        cat << EOF
status: "$DATASET_STATUS"
type: "$DATASET_TYPE"
path: "$DATASET_PATH"
config: "$DATASET_CONFIG"
counts:
  total: $TOTAL_COUNT
  train: $TRAIN_COUNT
  val: $VAL_COUNT
ready_for_training: $([ "$DATASET_STATUS" != "none" ] && echo "true" || echo "false")
EOF
        ;;
    *)
        log_error "Unknown output format: $OUTPUT_FORMAT"
        exit 1
        ;;
esac

# Exit with appropriate code
if [ "$DATASET_STATUS" = "none" ]; then
    exit 1
else
    exit 0
fi