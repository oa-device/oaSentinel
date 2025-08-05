#!/bin/bash

# oaSentinel Automated Training Script
# Designed for unattended training execution in production environments
# Usage: ./start_training_auto.sh [--config CONFIG_FILE] [--epochs EPOCHS]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="configs/default.yaml"
EPOCHS=""
DEVICE="auto"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# Logging functions
log_info() { 
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}
log_success() { 
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}
log_warning() { 
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}
log_error() { 
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}
log_header() { 
    echo -e "\n${CYAN}===== $1 =====${NC}" | tee -a "$LOG_FILE"
}

show_usage() {
    echo "oaSentinel Automated Training Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --config FILE     Training configuration file (default: configs/default.yaml)"
    echo "  --epochs N        Number of training epochs (overrides config)"
    echo "  --device DEVICE   Training device (default: auto)"
    echo "  -h, --help        Show this help message"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
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

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Start logging
log_header "oaSentinel Automated Training Session"
log_info "Started at: $(date)"
log_info "Project root: $PROJECT_ROOT"
log_info "Log file: $LOG_FILE"
log_info "Configuration: $CONFIG_FILE"

# Change to project directory
cd "$PROJECT_ROOT"

# Verify environment
log_header "Environment Verification"

if [ ! -f ".venv/bin/activate" ]; then
    log_error "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate
log_success "Virtual environment activated"

# Check Python packages
if ! python -c "import ultralytics, torch" 2>/dev/null; then
    log_error "Required packages not installed. Run ./setup.sh first."
    exit 1
fi
log_success "Required packages verified"

# Check GPU availability
GPU_INFO=$(python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null || echo "GPU check failed")
log_info "GPU Status: $GPU_INFO"

# Dataset detection and preparation
log_header "Dataset Preparation"

# Run dataset detection
if [ -f "scripts/detect_dataset.sh" ]; then
    DATASET_STATUS=$(./scripts/detect_dataset.sh --format summary 2>/dev/null || echo "Dataset detection failed")
    log_info "Dataset status: $DATASET_STATUS"
    
    # Get dataset info for training
    DATASET_INFO=$(./scripts/detect_dataset.sh --format env 2>/dev/null || true)
    if [ -n "$DATASET_INFO" ]; then
        eval "$DATASET_INFO"
        log_success "Dataset configuration loaded"
        
        # Use detected dataset YAML if available
        if [ -n "$DATASET_YAML" ] && [ -f "$DATASET_YAML" ]; then
            TRAINING_DATA="$DATASET_YAML"
            log_info "Using dataset: $TRAINING_DATA"
        else
            TRAINING_DATA="crowdhuman.yaml"
            log_warning "Using default dataset config: $TRAINING_DATA"
        fi
    else
        log_warning "Could not detect dataset, using default configuration"
        TRAINING_DATA="crowdhuman.yaml"
    fi
else
    log_warning "Dataset detection script not found, using default"
    TRAINING_DATA="crowdhuman.yaml"
fi

# Verify dataset file exists
if [ ! -f "$TRAINING_DATA" ]; then
    log_error "Dataset configuration not found: $TRAINING_DATA"
    log_info "Available dataset files:"
    find . -name "*.yaml" -type f | head -10 | tee -a "$LOG_FILE"
    exit 1
fi

# Training execution
log_header "Training Execution"

# Build training command
TRAIN_CMD="python train_yolo11m.py --data $TRAINING_DATA --device $DEVICE"

# Add epochs if specified
if [ -n "$EPOCHS" ]; then
    TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
fi

log_info "Training command: $TRAIN_CMD"

# Create training log symlink for easy access
ln -sf "$LOG_FILE" "$LOG_DIR/latest_training.log"

# Execute training with comprehensive logging
log_info "Starting training execution..."
echo "========== TRAINING OUTPUT ==========" >> "$LOG_FILE"

if eval "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"; then
    log_success "Training completed successfully!"
    
    # Check for output models
    if [ -d "runs/detect" ]; then
        LATEST_RUN=$(find runs/detect -name "train*" -type d | sort | tail -1)
        if [ -n "$LATEST_RUN" ]; then
            log_success "Training artifacts saved to: $LATEST_RUN"
            
            # Check for exported models
            if [ -f "$LATEST_RUN/weights/best.pt" ]; then
                log_success "Best model: $LATEST_RUN/weights/best.pt"
            fi
            
            # Run export if export script exists
            if [ -f "scripts/export.sh" ]; then
                log_info "Running model export..."
                if ./scripts/export.sh --model "$LATEST_RUN/weights/best.pt" --formats onnx coreml 2>&1 | tee -a "$LOG_FILE"; then
                    log_success "Model export completed"
                else
                    log_warning "Model export failed, but training was successful"
                fi
            fi
        fi
    fi
    
    TRAINING_STATUS="SUCCESS"
else
    log_error "Training failed!"
    TRAINING_STATUS="FAILED"
fi

# Final summary
log_header "Training Session Summary"
log_info "Status: $TRAINING_STATUS"
log_info "Completed at: $(date)"
log_info "Duration: $(date -d@$(($(date +%s) - $(stat -c %Y "$LOG_FILE"))) -u +%H:%M:%S)"
log_info "Log file: $LOG_FILE"

# Create status file for monitoring
echo "$TRAINING_STATUS" > "$LOG_DIR/last_training_status.txt"
echo "$(date)" >> "$LOG_DIR/last_training_status.txt"
echo "$LOG_FILE" >> "$LOG_DIR/last_training_status.txt"

if [ "$TRAINING_STATUS" = "SUCCESS" ]; then
    log_success "🎉 Training automation completed successfully!"
    exit 0
else
    log_error "❌ Training automation failed!"
    exit 1
fi
