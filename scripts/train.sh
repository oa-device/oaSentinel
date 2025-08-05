#!/bin/bash

# oaSentinel Training Script
# Train YOLO models for human detection optimization
# Usage: ./scripts/train.sh [--config path] [--model arch] [--epochs num]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="configs/default.yaml"
MODEL_ARCH="yolo11m"  # Default to YOLO11m as specified in requirements
EPOCHS="100"          # Default to 100 epochs as specified in requirements
DEVICE="auto"
WANDB_MODE="disabled"
RESUME=""

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${BLUE}===== $1 =====${NC}"; }

show_usage() {
    echo "oaSentinel Training Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --config FILE     Training configuration file (default: configs/default.yaml)"
    echo "  --model ARCH      Model architecture (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x) [default: yolo11m]"
    echo "  --epochs NUM      Number of training epochs [default: 100]"
    echo "  --device DEVICE   Training device (auto/cpu/gpu/0/1/[0,1]/...)"
    echo "  --wandb          Enable Weights & Biases experiment tracking"
    echo "  --resume PATH     Resume training from checkpoint"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Train with default config (yolo11m, 100 epochs)"
    echo "  $0 --model yolo11s --epochs 50       # Quick training with small model"
    echo "  $0 --device '[0,1]' --epochs 100     # Dual-GPU training with RTX GPUs"
    echo "  $0 --config configs/custom.yaml      # Train with custom config"
    echo "  $0 --wandb                           # Enable experiment tracking"
    echo "  $0 --resume models/checkpoints/last.pt  # Resume training"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_ARCH="$2"
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
        --wandb)
            WANDB_MODE="online"
            shift
            ;;
        --resume)
            RESUME="$2"
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

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

log_header "oaSentinel Model Training"
log_info "Configuration: $CONFIG_FILE"
log_info "Device: $DEVICE"
log_info "Experiment tracking: $WANDB_MODE"

# Create necessary directories
mkdir -p models/checkpoints
mkdir -p logs/training
mkdir -p outputs/visualizations

# Generate unique run name
RUN_NAME="training_$(date +%Y%m%d_%H%M%S)"
if [ -n "$MODEL_ARCH" ]; then
    RUN_NAME="${RUN_NAME}_${MODEL_ARCH}"
fi

log_info "Run name: $RUN_NAME"

# Set up Weights & Biases if enabled
if [ "$WANDB_MODE" = "online" ]; then
    if [ -z "$WANDB_API_KEY" ]; then
        log_warning "WANDB_API_KEY not set. Set it in .env file or environment"
        log_info "You can get your API key from: https://wandb.ai/authorize"
        WANDB_MODE="disabled"
    else
        log_info "Weights & Biases tracking enabled"
        export WANDB_PROJECT="${WANDB_PROJECT:-oaSentinel}"
        export WANDB_NAME="$RUN_NAME"
    fi
fi

export WANDB_MODE="$WANDB_MODE"

# Start training with Python
log_info "Starting training process..."

python3 << 'EOF'
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import wandb

def load_config(config_path):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_wandb(config, run_name):
    """Initialize Weights & Biases if enabled"""
    if os.environ.get('WANDB_MODE') == 'online':
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'oaSentinel'),
            name=run_name,
            config=config,
            tags=config.get('experiment', {}).get('tags', [])
        )
        return True
    return False

def main():
    # Configuration
    config_file = os.environ.get('CONFIG_FILE', 'configs/default.yaml')
    model_arch = os.environ.get('MODEL_ARCH', '')
    epochs_override = os.environ.get('EPOCHS', '')
    device = os.environ.get('DEVICE', 'auto')
    resume_path = os.environ.get('RESUME', '')
    run_name = os.environ.get('RUN_NAME', 'training')
    
    print(f"Loading configuration from: {config_file}")
    config = load_config(config_file)
    
    # Override config with command line arguments
    if model_arch:
        config['model']['architecture'] = model_arch
        print(f"Model architecture overridden: {model_arch}")
    
    if epochs_override:
        config['training']['epochs'] = int(epochs_override)
        print(f"Epochs overridden: {epochs_override}")
    
    # Setup experiment tracking
    wandb_enabled = setup_wandb(config, run_name)
    
    # Initialize model
    model_name = config['model']['architecture']
    pretrained = config['model'].get('pretrained', True)
    
    print(f"Initializing {model_name} model (pretrained={pretrained})")
    
    if resume_path and Path(resume_path).exists():
        print(f"Resuming training from: {resume_path}")
        model = YOLO(resume_path)
    else:
        if pretrained:
            model = YOLO(f"{model_name}.pt")  # Load pretrained weights
        else:
            model = YOLO(f"{model_name}.yaml")  # Load architecture only
    
    # Get dataset configuration
    dataset_config = config.get('dataset', {})
    dataset_name = dataset_config.get('name', 'crowdhuman')
    dataset_path = dataset_config.get('path', f'data/processed/{dataset_name}')
    
    # Look for dataset YAML file with different naming patterns
    dataset_yaml_candidates = [
        "crowdhuman.yaml",  # User-specified crowdhuman.yaml in root
        f"{dataset_path}/dataset.yaml",
        f"{dataset_path}/{dataset_name}.yaml",
        f"data/splits/{dataset_name}/dataset.yaml",
        f"data/{dataset_name}/dataset.yaml"
    ]
    
    dataset_yaml = None
    for candidate in dataset_yaml_candidates:
        if Path(candidate).exists():
            dataset_yaml = candidate
            break
    
    if not dataset_yaml:
        print(f"ERROR: Dataset YAML not found. Tried: {dataset_yaml_candidates}")
        sys.exit(1)
    
    print(f"Using dataset configuration: {dataset_yaml}")
    
    # Training parameters
    training_config = config.get('training', {})
    
    train_args = {
        'data': dataset_yaml,
        'epochs': training_config.get('epochs', 100),
        'batch': dataset_config.get('batch_size', training_config.get('batch_size', 16)),
        'imgsz': dataset_config.get('image_size', 640),
        'device': device,
        'workers': training_config.get('workers', 4),
        'project': 'models/checkpoints',
        'name': run_name,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'patience': training_config.get('patience', 10),
        'lr0': training_config.get('learning_rate', 0.001),
        'optimizer': training_config.get('optimizer', 'AdamW'),
        'cos_lr': training_config.get('scheduler') == 'cosine',
        'cache': True,  # Cache images for faster training
        'val': True,    # Validate during training
    }
    
    # Add augmentation parameters
    augment_config = training_config.get('augment', {})
    for key, value in augment_config.items():
        train_args[key] = value
    
    print(f"\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Check GPU availability and setup device
    print(f"\nDevice setup:")
    print(f"  Requested device: {device}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if device == 'auto':
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 2:
                device = [0, 1]  # Use dual-GPU as specified in requirements
                print(f"Auto-detected dual-GPU setup: Using GPUs {device}")
            else:
                device = 0  # Use first GPU
                print(f"Single GPU detected: Using GPU {device}")
        else:
            device = 'cpu'
            print("CUDA not available: Using CPU")
        train_args['device'] = device
    else:
        # Handle string representation of device list like '[0,1]'
        if isinstance(device, str) and device.startswith('[') and device.endswith(']'):
            try:
                device = eval(device)  # Convert '[0,1]' to [0,1]
                print(f"Using specified multi-GPU setup: {device}")
            except:
                print(f"Warning: Could not parse device list {device}, falling back to auto")
                device = 'auto'
        train_args['device'] = device
    
    print(f"\nStarting training on device: {device}")
    print(f"Experiment tracking: {'enabled' if wandb_enabled else 'disabled'}")
    print("-" * 50)
    
    try:
        # Start training
        results = model.train(**train_args)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        # Print key metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"Best mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"Best mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        
        # Model paths
        model_dir = Path(f"models/checkpoints/{run_name}")
        print(f"\nModel files saved to: {model_dir}")
        print(f"Best model: {model_dir}/weights/best.pt")
        print(f"Last model: {model_dir}/weights/last.pt")
        
        # Log completion to wandb
        if wandb_enabled:
            wandb.log({"training_status": "completed"})
            wandb.finish()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if wandb_enabled:
            wandb.log({"training_status": "interrupted"})
            wandb.finish()
        sys.exit(130)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if wandb_enabled:
            wandb.log({"training_status": "failed", "error": str(e)})
            wandb.finish()
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Set environment variables for Python script
export CONFIG_FILE="$CONFIG_FILE"
export MODEL_ARCH="$MODEL_ARCH"
export EPOCHS="$EPOCHS"
export DEVICE="$DEVICE"
export RESUME="$RESUME"
export RUN_NAME="$RUN_NAME"

# Check exit status
if [ $? -eq 0 ]; then
    log_header "Training Completed Successfully!"
    log_success "Model checkpoints saved to: models/checkpoints/$RUN_NAME/"
    log_info "Best model: models/checkpoints/$RUN_NAME/weights/best.pt"
    log_info "Last model: models/checkpoints/$RUN_NAME/weights/last.pt"
    echo ""
    log_info "Next steps:"
    log_info "1. Evaluate the model: ./scripts/evaluate.sh --model models/checkpoints/$RUN_NAME/weights/best.pt"
    log_info "2. Export for deployment: ./scripts/export.sh --model models/checkpoints/$RUN_NAME/weights/best.pt"
else
    log_error "Training failed. Check the logs above for details."
    exit 1
fi