#!/bin/bash

# oaSentinel Data Download Script
# Downloads and prepares the CrowdHuman dataset for training
# Usage: ./scripts/download_data.sh [--dataset crowdhuman] [--force]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/raw"
DATASET="crowdhuman"
FORCE_DOWNLOAD=false

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
        --force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        -h|--help)
            echo "oaSentinel Data Download Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dataset NAME  Dataset to download (default: crowdhuman)"
            echo "  --force        Force re-download even if data exists"
            echo "  -h, --help     Show this help message"
            echo ""
            echo "Supported datasets:"
            echo "  crowdhuman     CrowdHuman dataset (human detection)"
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

log_header "oaSentinel Data Download"
log_info "Dataset: $DATASET"
log_info "Data directory: $DATA_DIR"

# Create data directories
mkdir -p "$DATA_DIR/$DATASET"

# Download CrowdHuman dataset
download_crowdhuman() {
    local dataset_dir="$DATA_DIR/crowdhuman"
    
    log_info "Downloading CrowdHuman dataset..."
    
    # Check if dataset already exists
    if [ -d "$dataset_dir" ] && [ -f "$dataset_dir/annotation_train.odgt" ] && [ "$FORCE_DOWNLOAD" = false ]; then
        log_warning "CrowdHuman dataset already exists. Use --force to re-download."
        return 0
    fi
    
    # CrowdHuman URLs
    local base_url="https://www.crowdhuman.org/download"
    local files=(
        "annotation_train.odgt"
        "annotation_val.odgt"
        "Images.zip"
    )
    
    # Note: CrowdHuman requires registration and manual download
    # This script provides instructions and validates the files
    
    log_warning "CrowdHuman dataset requires manual registration and download."
    echo ""
    echo "Please follow these steps:"
    echo "1. Visit: https://www.crowdhuman.org/"
    echo "2. Register and agree to the terms of use"
    echo "3. Download the following files to $dataset_dir:"
    echo "   📄 annotation_train.odgt (~47MB)"
    echo "   📄 annotation_val.odgt (~14MB)" 
    echo "   📦 Images.zip (~15GB) or extract to Images/ folder"
    echo ""
    echo "💡 Citation required:"
    echo "@article{shao2018crowdhuman,"
    echo "    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},"
    echo "    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},"
    echo "    journal={arXiv preprint arXiv:1805.00123},"
    echo "    year={2018}"
    echo "}"
    echo ""
    echo "Expected directory structure:"
    echo "$dataset_dir/"
    echo "├── annotation_train.odgt"
    echo "├── annotation_val.odgt"
    echo "└── Images/"
    echo "    ├── 273271,1a0d6000b9e1f5b7.jpg"
    echo "    ├── 273271,1a10d000b9e1f5b7.jpg"
    echo "    └── ... (~24,370 total images)"
    echo ""
    
    # Wait for user confirmation with timeout
    echo "⏳ Waiting for manual download completion..."
    read -t 300 -p "Press Enter when you have downloaded and extracted the files (or wait 5 minutes to continue): " || echo ""
    
    # Validate downloaded files
    log_info "Validating downloaded files..."
    
    local required_files=(
        "$dataset_dir/annotation_train.odgt"
        "$dataset_dir/annotation_val.odgt"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file not found: $file"
            return 1
        fi
    done
    
    # Check for Images directory or zip file
    if [ ! -d "$dataset_dir/Images" ] && [ ! -f "$dataset_dir/Images.zip" ]; then
        log_error "Images directory or Images.zip not found"
        return 1
    fi
    
    # Extract images if zip exists
    if [ -f "$dataset_dir/Images.zip" ]; then
        log_info "Extracting Images.zip..."
        cd "$dataset_dir"
        unzip -q Images.zip
        rm Images.zip
        cd "$PROJECT_ROOT"
        log_success "Images extracted"
    fi
    
    # Validate image count and file sizes
    local image_count=$(find "$dataset_dir/Images" -name "*.jpg" 2>/dev/null | wc -l)
    log_info "Found $image_count image files"
    
    # Check annotation file sizes (approximate)
    local train_size=$(stat -f%z "$dataset_dir/annotation_train.odgt" 2>/dev/null || echo "0")
    local val_size=$(stat -f%z "$dataset_dir/annotation_val.odgt" 2>/dev/null || echo "0")
    
    log_info "Annotation file sizes:"
    log_info "  annotation_train.odgt: $(( train_size / 1024 / 1024 ))MB"
    log_info "  annotation_val.odgt: $(( val_size / 1024 / 1024 ))MB"
    
    # Validation checks
    local validation_errors=0
    
    if [ "$image_count" -lt 20000 ]; then
        log_warning "Expected ~24,370 images, found $image_count. Dataset may be incomplete."
        ((validation_errors++))
    fi
    
    if [ "$train_size" -lt 40000000 ]; then  # ~40MB minimum
        log_warning "annotation_train.odgt seems small ($(( train_size / 1024 / 1024 ))MB). Expected ~47MB."
        ((validation_errors++))
    fi
    
    if [ "$val_size" -lt 10000000 ]; then  # ~10MB minimum
        log_warning "annotation_val.odgt seems small ($(( val_size / 1024 / 1024 ))MB). Expected ~14MB."
        ((validation_errors++))
    fi
    
    if [ "$validation_errors" -eq 0 ]; then
        log_success "CrowdHuman dataset validated successfully"
    else
        log_warning "Dataset validation completed with $validation_errors warnings"
        log_info "You can proceed, but the dataset might be incomplete"
    fi
    
    # Create dataset info file
    cat > "$dataset_dir/dataset_info.yaml" << EOF
# CrowdHuman Dataset Information
name: CrowdHuman
version: "1.0"
description: "A benchmark dataset to better evaluate detectors in crowd scenarios"
source: "https://www.crowdhuman.org/"
license: "Academic use only"

statistics:
  total_images: $image_count
  train_images: $(wc -l < "$dataset_dir/annotation_train.odgt")
  val_images: $(wc -l < "$dataset_dir/annotation_val.odgt")
  classes: ["person"]
  
download_date: "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
downloaded_by: "$(whoami)"

files:
  annotations:
    train: "annotation_train.odgt"
    validation: "annotation_val.odgt"
  images: "Images/"
  
format: "ODGT (Object Detection Ground Truth)"
note: "Manual download required from official website"
EOF
    
    log_success "Dataset info file created"
}

# Main download logic
case $DATASET in
    "crowdhuman")
        download_crowdhuman
        ;;
    *)
        log_error "Unknown dataset: $DATASET"
        log_info "Supported datasets: crowdhuman"
        exit 1
        ;;
esac

log_header "Download Complete"
log_success "Dataset '$DATASET' is ready for processing"
log_info "Next step: ./scripts/process_data.sh --dataset $DATASET"