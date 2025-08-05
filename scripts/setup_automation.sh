#!/bin/bash

# oaSentinel Automation Setup Script
# One-stop setup script optimized for automation environments
# Usage: ./scripts/setup_automation.sh [--test-mode] [--force-download] [--skip-data]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_MODE=false
FORCE_DOWNLOAD=false
SKIP_DATA=false

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${CYAN}===== $1 =====${NC}"; }

show_usage() {
    echo "oaSentinel Automation Setup Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --test-mode       Use test dataset for quick setup"
    echo "  --force-download  Force re-download of datasets"
    echo "  --skip-data       Skip data download/processing"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "This script provides a complete, automation-friendly setup of oaSentinel."
    echo "It handles environment setup, dataset management, and configuration."
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        --force-download)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
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

log_header "oaSentinel Automation Setup"
log_info "Project directory: $PROJECT_ROOT"
log_info "Test mode: $TEST_MODE"
log_info "Skip data: $SKIP_DATA"

# Environment validation
log_header "Environment Validation"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src" ]; then
    log_error "Not in oaSentinel project root. Please run from project directory."
    exit 1
fi

# Check Python environment
if [ ! -f ".venv/bin/activate" ]; then
    log_warning "Virtual environment not found"
    if [ -f "setup.sh" ]; then
        log_info "Running setup.sh to create environment..."
        ./setup.sh
    else
        log_info "Creating virtual environment..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip uv
        
        if [ -f "pyproject.toml" ]; then
            uv pip install -e .
        fi
    fi
else
    log_success "Virtual environment found"
fi

# Activate virtual environment
source .venv/bin/activate
log_success "Virtual environment activated"

# Verify core dependencies
log_info "Verifying core dependencies..."
python3 -c "
import sys
required_packages = ['torch', 'ultralytics', 'cv2', 'yaml']
missing = []

for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('✅ All core dependencies available')
" || {
    log_error "Missing core dependencies"
    log_info "Installing dependencies..."
    pip install --upgrade pip uv
    if [ -f "pyproject.toml" ]; then
        uv pip install -e .
    else
        uv pip install torch torchvision ultralytics opencv-python pyyaml typer rich
    fi
}

# Dataset management
if [ "$SKIP_DATA" = false ]; then
    log_header "Dataset Detection and Setup"
    
    # Use our smart dataset detection
    DATASET_INFO=""
    if [ -f "scripts/detect_dataset.sh" ]; then
        if DATASET_INFO=$(./scripts/detect_dataset.sh --format env 2>/dev/null); then
            # Source the dataset info
            eval "$DATASET_INFO"
            log_success "Dataset detection completed"
            log_info "Status: $DATASET_STATUS"
            log_info "Type: $DATASET_TYPE"
            log_info "Total images: $DATASET_TOTAL_COUNT"
        else
            log_warning "No dataset detected"
            DATASET_STATUS="none"
        fi
    else
        log_warning "Dataset detection script not found, checking manually..."
        DATASET_STATUS="none"
    fi
    
    # Handle dataset setup based on mode and current status
    if [ "$TEST_MODE" = true ]; then
        log_info "Test mode enabled - ensuring test dataset is available"
        
        if [ "$DATASET_TYPE" != "test_dataset" ] || [ "$DATASET_TOTAL_COUNT" -eq 0 ]; then
            log_info "Creating/updating test dataset..."
            if [ -f "scripts/create_test_dataset.sh" ]; then
                ./scripts/create_test_dataset.sh --samples 20
            else
                log_warning "Test dataset creation script not found, creating minimal dataset..."
                mkdir -p data/test_dataset/{images/{train,val},labels/{train,val}}
                
                # Create minimal dataset files
                for i in {0..3}; do
                    # Create simple test images (1x1 pixel JPEGs)
                    printf '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' ",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9' > "data/test_dataset/images/train/train_$(printf "%04d" $i).jpg"
                    
                    # Create corresponding labels
                    echo "0 0.5 0.5 0.3 0.6" > "data/test_dataset/labels/train/train_$(printf "%04d" $i).txt"
                done
                
                # Create validation images
                for i in {0..1}; do
                    printf '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' ",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9' > "data/test_dataset/images/val/val_$(printf "%04d" $i).jpg"
                    
                    echo "0 0.5 0.5 0.3 0.6" > "data/test_dataset/labels/val/val_$(printf "%04d" $i).txt"
                done
                
                # Create dataset config
                cat > data/test_dataset/dataset.yaml << EOF
path: $(pwd)/data/test_dataset
train: images/train
val: images/val

nc: 1
names: ['person']

info:
  name: "oaSentinel Test Dataset"
  description: "Minimal synthetic dataset for automation testing"
  created: "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  samples: 6
  format: "YOLO"
  synthetic: true
EOF
            fi
            log_success "Test dataset ready"
        else
            log_success "Test dataset already available"
        fi
        
    else
        # Production mode - try to get real dataset
        log_info "Production mode - attempting to download full dataset"
        
        if [ "$DATASET_STATUS" = "none" ] || [ "$FORCE_DOWNLOAD" = true ]; then
            log_info "Downloading CrowdHuman dataset..."
            if [ -f "scripts/download_data.sh" ]; then
                ./scripts/download_data.sh || {
                    log_warning "Dataset download failed, falling back to test mode"
                    TEST_MODE=true
                }
            else
                log_warning "Download script not found, enabling test mode"
                TEST_MODE=true
            fi
        else
            log_success "Dataset already available: $DATASET_TYPE"
        fi
        
        # If we fell back to test mode, ensure test dataset exists
        if [ "$TEST_MODE" = true ]; then
            log_info "Setting up test dataset as fallback..."
            if [ -f "scripts/create_test_dataset.sh" ]; then
                ./scripts/create_test_dataset.sh --samples 20
            fi
        fi
    fi
    
    # Process dataset if needed
    log_header "Dataset Processing"
    
    # Re-detect dataset after potential creation/download
    if [ -f "scripts/detect_dataset.sh" ]; then
        if DATASET_INFO=$(./scripts/detect_dataset.sh --format env 2>/dev/null); then
            eval "$DATASET_INFO"
        fi
    fi
    
    if [ "$DATASET_STATUS" = "ready" ] && [ "$DATASET_TYPE" = "crowdhuman_full" ]; then
        log_info "Processing full CrowdHuman dataset..."
        if [ -f "scripts/process_data.sh" ]; then
            ./scripts/process_data.sh || {
                log_warning "Dataset processing failed, but continuing with raw data"
            }
        fi
    elif [ "$DATASET_STATUS" = "test" ]; then
        log_info "Test dataset detected - no processing needed"
    else
        log_warning "No suitable dataset found for processing"
    fi
fi

# Configuration setup
log_header "Configuration Setup"

# Update dataset configuration based on what we found
if [ -f "scripts/detect_dataset.sh" ]; then
    if DATASET_INFO=$(./scripts/detect_dataset.sh --format env 2>/dev/null); then
        eval "$DATASET_INFO"
        
        # Create or update crowdhuman.yaml to point to detected dataset
        if [ "$DATASET_STATUS" = "test" ]; then
            log_info "Configuring for test dataset..."
            
            # Create crowdhuman.yaml pointing to test dataset
            cat > crowdhuman.yaml << EOF
# oaSentinel Dataset Configuration (Test Mode)
path: data/test_dataset
train: images/train
val: images/val

nc: 1
names:
  0: person

# Test dataset configuration
test_mode: true
automation_ready: true
EOF
            log_success "Test configuration created"
            
        elif [ "$DATASET_STATUS" = "ready" ] || [ "$DATASET_STATUS" = "processed" ]; then
            log_info "Configuring for production dataset..."
            
            # Ensure crowdhuman.yaml exists and is properly configured
            if [ ! -f "crowdhuman.yaml" ] || [ "$FORCE_DOWNLOAD" = true ]; then
                cat > crowdhuman.yaml << EOF
# oaSentinel Dataset Configuration (Production Mode)
path: .
train: images/train
val: images/val

nc: 2
names:
  0: person
  1: head

# Production dataset configuration
test_mode: false
automation_ready: true
EOF
            fi
            log_success "Production configuration ready"
        fi
    fi
fi

# Environment summary
log_header "Setup Summary"

# Final validation
if [ -f "scripts/detect_dataset.sh" ]; then
    FINAL_DATASET_INFO=$(./scripts/detect_dataset.sh --format env 2>/dev/null) || true
    if [ -n "$FINAL_DATASET_INFO" ]; then
        eval "$FINAL_DATASET_INFO"
        
        log_success "🎯 oaSentinel Automation Setup Complete!"
        echo ""
        log_info "📊 Dataset Status:"
        log_info "  Type: $DATASET_TYPE"
        log_info "  Status: $DATASET_STATUS"
        log_info "  Total Images: $DATASET_TOTAL_COUNT"
        log_info "  Training Images: $DATASET_TRAIN_COUNT"
        log_info "  Validation Images: $DATASET_VAL_COUNT"
        log_info "  Config: $DATASET_CONFIG"
        echo ""
        log_info "🚀 Ready for Training:"
        if [ "$TEST_MODE" = true ] || [ "$DATASET_TYPE" = "test_dataset" ]; then
            log_info "  ./scripts/train.sh --config configs/default.yaml --epochs 5"
        else
            log_info "  ./scripts/train.sh --config configs/default.yaml"
        fi
        echo ""
        log_info "🎮 Quick Commands:"
        log_info "  ./scripts/train.sh                    # Start training"
        log_info "  ./scripts/detect_dataset.sh --verbose # Check dataset"
        log_info "  source .venv/bin/activate             # Activate environment"
        
        # Set automation-friendly exit code
        exit 0
    else
        log_error "Dataset detection failed after setup"
        exit 1
    fi
else
    log_warning "Dataset detection not available, basic setup completed"
    log_success "Environment setup completed"
    exit 0
fi