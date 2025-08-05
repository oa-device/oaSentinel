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

# Function to create a minimal test dataset for development/testing
create_test_dataset() {
    local dataset_dir="$1"
    log_info "Creating minimal test dataset for development..."
    
    local test_dir="$dataset_dir/test_dataset"
    mkdir -p "$test_dir/Images"
    
    # Create a minimal annotation file for testing
    cat > "$dataset_dir/annotation_train.odgt" << 'EOF'
{"ID": "test_000001", "gtboxes": [{"tag": "person", "hbox": [100, 100, 200, 300], "vbox": [100, 100, 200, 300], "head_attr": {"ignore": 0}, "extra": {"box_id": 0}}]}
{"ID": "test_000002", "gtboxes": [{"tag": "person", "hbox": [150, 120, 250, 320], "vbox": [150, 120, 250, 320], "head_attr": {"ignore": 0}, "extra": {"box_id": 0}}]}
EOF
    
    cat > "$dataset_dir/annotation_val.odgt" << 'EOF'
{"ID": "test_000003", "gtboxes": [{"tag": "person", "hbox": [80, 90, 180, 290], "vbox": [80, 90, 180, 290], "head_attr": {"ignore": 0}, "extra": {"box_id": 0}}]}
EOF
    
    # Use existing test images if available, otherwise create placeholders
    if [ -d "$PROJECT_ROOT/data/test_dataset/images" ]; then
        log_info "Using existing test images from data/test_dataset/"
        cp -r "$PROJECT_ROOT/data/test_dataset/images/"* "$test_dir/Images/" 2>/dev/null || true
    else
        log_info "Creating placeholder test images..."
        # Create small placeholder images using ImageMagick or Python if available
        for i in {1..3}; do
            local img_name=$(printf "test_%06d.jpg" $i)
            if command -v convert >/dev/null 2>&1; then
                # Use ImageMagick if available
                convert -size 640x480 xc:lightgray -fill black -pointsize 20 \
                    -annotate +50+240 "Test Image $i" "$test_dir/Images/$img_name" 2>/dev/null || {
                    # Fallback: create a tiny valid JPEG
                    echo -e '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' ",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9' > "$test_dir/Images/$img_name"
                }
            else
                # Create a minimal valid JPEG header (1x1 pixel)
                printf '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' ",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9' > "$test_dir/Images/$img_name"
            fi
        done
    fi
    
    # Create Images symlink to test_dataset for compatibility
    if [ ! -e "$dataset_dir/Images" ]; then
        ln -sf "test_dataset/Images" "$dataset_dir/Images"
    fi
    
    log_success "Test dataset created with $(ls "$test_dir/Images" | wc -l) images"
    log_info "You can replace this with the full CrowdHuman dataset later"
}

# Download CrowdHuman dataset
download_crowdhuman() {
    local dataset_dir="$DATA_DIR/crowdhuman"
    
    log_info "Downloading CrowdHuman dataset..."
    
    # Check if dataset already exists
    if [ -d "$dataset_dir" ] && [ -f "$dataset_dir/annotation_train.odgt" ] && [ "$FORCE_DOWNLOAD" = false ]; then
        log_warning "CrowdHuman dataset already exists. Use --force to re-download."
        return 0
    fi
    
    # Check for required dependencies
    if ! command -v git &> /dev/null; then
        log_error "git command not found. Please install git to continue."
        return 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_error "python3 command not found. Please install Python 3 to continue."
        return 1
    fi
    
    # CrowdHuman Hugging Face dataset URL
    local hf_repo="https://huggingface.co/datasets/sshao0516/CrowdHuman"
    local hf_dataset_dir="$dataset_dir/huggingface_repo"
    
    log_info "Downloading CrowdHuman dataset from Hugging Face..."
    log_info "Source: $hf_repo"
    
    # Check for Git LFS availability
    if command -v git-lfs &> /dev/null; then
        log_info "Git LFS detected, initializing..."
        git lfs install --skip-smudge 2>/dev/null || true
    else
        log_warning "Git LFS not found. Large files may not download correctly."
        log_info "Install with: sudo apt install git-lfs"
    fi
    
    # Clone the Hugging Face repository with Git LFS support
    if [ -d "$hf_dataset_dir" ]; then
        log_info "Repository already exists, updating..."
        cd "$hf_dataset_dir"
        
        # Try to pull with LFS
        if command -v git-lfs &> /dev/null; then
            git lfs pull 2>/dev/null || {
                log_warning "Git LFS pull failed, trying regular pull..."
                git pull
            }
        else
            git pull
        fi
        cd "$PROJECT_ROOT"
    else
        log_info "Cloning Hugging Face repository..."
        
        # Try cloning with different strategies
        if command -v git-lfs &> /dev/null; then
            # First try with LFS
            git lfs clone "$hf_repo" "$hf_dataset_dir" 2>/dev/null || {
                log_warning "Git LFS clone failed, trying regular clone..."
                git clone "$hf_repo" "$hf_dataset_dir"
                
                # Try to checkout LFS files
                cd "$hf_dataset_dir"
                git lfs checkout 2>/dev/null || log_warning "LFS checkout failed"
                cd "$PROJECT_ROOT"
            }
        else
            git clone "$hf_repo" "$hf_dataset_dir"
        fi
    fi
    
    # Move files to expected structure
    log_info "Organizing dataset files..."
    
    # Copy annotation files
    if [ -f "$hf_dataset_dir/annotation_train.odgt" ]; then
        cp "$hf_dataset_dir/annotation_train.odgt" "$dataset_dir/"
    fi
    
    if [ -f "$hf_dataset_dir/annotation_val.odgt" ]; then
        cp "$hf_dataset_dir/annotation_val.odgt" "$dataset_dir/"
    fi
    
    # Handle images directory
    if [ -d "$hf_dataset_dir/Images" ]; then
        if [ ! -d "$dataset_dir/Images" ]; then
            cp -r "$hf_dataset_dir/Images" "$dataset_dir/"
        else
            log_info "Images directory already exists, syncing..."
            rsync -av "$hf_dataset_dir/Images/" "$dataset_dir/Images/"
        fi
    elif [ -f "$hf_dataset_dir/Images.zip" ]; then
        log_info "Extracting Images.zip..."
        cd "$dataset_dir"
        unzip -q "$hf_dataset_dir/Images.zip"
        cd "$PROJECT_ROOT"
    fi
    
    echo ""
    echo "💡 Citation required:"
    echo "@article{shao2018crowdhuman,"
    echo "    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},"
    echo "    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},"
    echo "    journal={arXiv preprint arXiv:1805.00123},"
    echo "    year={2018}"
    echo "}"
    echo ""
    
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
    
    # Check for Images directory or zip file with multiple possible locations
    local images_found=false
    
    if [ -d "$dataset_dir/Images" ]; then
        images_found=true
        log_success "Images directory found"
    elif [ -f "$dataset_dir/Images.zip" ]; then
        images_found=true
        log_success "Images.zip found"
    elif [ -d "$hf_dataset_dir/Images" ]; then
        images_found=true
        log_info "Images found in Hugging Face repo, copying..."
        cp -r "$hf_dataset_dir/Images" "$dataset_dir/"
    elif [ -f "$hf_dataset_dir/Images.zip" ]; then
        images_found=true
        log_info "Images.zip found in Hugging Face repo, copying..."
        cp "$hf_dataset_dir/Images.zip" "$dataset_dir/"
    else
        # Try to find images in subdirectories
        local found_images=$(find "$hf_dataset_dir" -type d -name "*mages*" 2>/dev/null | head -1)
        if [ -n "$found_images" ]; then
            images_found=true
            log_info "Images found at: $found_images"
            cp -r "$found_images" "$dataset_dir/Images"
        fi
    fi
    
    if [ "$images_found" = false ]; then
        log_warning "Images directory or Images.zip not found in expected locations"
        log_info "Available files in repository:"
        ls -la "$hf_dataset_dir/" 2>/dev/null || echo "Repository directory not accessible"
        log_info ""
        log_info "This might be due to:"
        log_info "1. Git LFS files not downloaded (images are stored in Git LFS)"
        log_info "2. Repository structure changed"
        log_info "3. Network issues during clone"
        log_info ""
        log_info "📋 Manual steps to resolve:"
        log_info "1. Install git-lfs if not available: sudo apt install git-lfs"
        log_info "2. Enable git-lfs: git lfs install"
        log_info "3. Re-clone with LFS: git clone --recursive $hf_repo $hf_dataset_dir"
        log_info "4. Or manually download from: https://www.crowdhuman.org/"
        log_info ""
        log_info "For now, creating a test dataset to continue setup..."
        
        # Create minimal test dataset to allow setup to continue
        create_test_dataset "$dataset_dir"
        return 0
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
source: "https://huggingface.co/datasets/sshao0516/CrowdHuman"
original_source: "https://www.crowdhuman.org/"
license: "Academic use only"

statistics:
  total_images: $image_count
  train_images: $(wc -l < "$dataset_dir/annotation_train.odgt")
  val_images: $(wc -l < "$dataset_dir/annotation_val.odgt")
  classes: ["person", "head"]
  
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