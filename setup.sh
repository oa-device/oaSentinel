#!/bin/bash

# oaSentinel Setup Script
# Automated environment setup for oaSentinel AI model development
# Usage: ./setup.sh [--clean] [--gpu] [--dev]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
VENV_NAME=".venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${BLUE}===== $1 =====${NC}"
}

# Parse command line arguments
CLEAN_INSTALL=false
GPU_SUPPORT=false
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_INSTALL=true
            shift
            ;;
        --gpu)
            GPU_SUPPORT=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        -h|--help)
            echo "oaSentinel Setup Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean    Clean install (remove existing environment)"
            echo "  --gpu      Install GPU support (CUDA)"
            echo "  --dev      Install development dependencies"
            echo "  -h, --help Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                # Basic installation"
            echo "  $0 --dev          # Development setup"
            echo "  $0 --clean --gpu  # Clean install with GPU support"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            log_info "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

log_header "oaSentinel Environment Setup"
log_info "Project root: $PROJECT_ROOT"
log_info "Python version: $PYTHON_VERSION"
log_info "Virtual environment: $VENV_NAME"

# Check system requirements
log_header "Checking System Requirements"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    log_warning "UV package manager not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Reload shell to pick up UV
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        log_error "Failed to install UV. Please install manually: https://github.com/astral-sh/uv"
        exit 1
    fi
fi

log_success "UV package manager found: $(uv --version)"

# Check Python version availability
if ! uv python find $PYTHON_VERSION &> /dev/null; then
    log_warning "Python $PYTHON_VERSION not found. Installing via UV..."
    uv python install $PYTHON_VERSION
fi

log_success "Python $PYTHON_VERSION available"

# Clean installation if requested
if [ "$CLEAN_INSTALL" = true ]; then
    log_header "Cleaning Previous Installation"
    if [ -d "$VENV_NAME" ]; then
        log_info "Removing existing virtual environment..."
        rm -rf "$VENV_NAME"
        log_success "Virtual environment removed"
    fi
    
    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    log_success "Python cache cleaned"
fi

# Create virtual environment
log_header "Setting Up Virtual Environment"

if [ ! -d "$VENV_NAME" ]; then
    log_info "Creating virtual environment with Python $PYTHON_VERSION..."
    uv venv --python $PYTHON_VERSION "$VENV_NAME"
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip and install core dependencies
log_header "Installing Dependencies"

log_info "Installing core project dependencies..."
uv pip install -e .

# Install optional dependencies
if [ "$GPU_SUPPORT" = true ]; then
    log_info "Installing GPU support dependencies..."
    uv pip install -e ".[gpu]"
    log_success "GPU support installed"
fi

if [ "$DEV_MODE" = true ]; then
    log_info "Installing development dependencies..."
    uv pip install -e ".[dev]"
    log_success "Development dependencies installed"
fi

# Setup Git hooks if in development mode
if [ "$DEV_MODE" = true ]; then
    log_header "Setting Up Development Environment"
    
    # Install pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        log_info "Setting up pre-commit hooks..."
        pre-commit install
        log_success "Pre-commit hooks installed"
    else
        log_warning "pre-commit not found, skipping hooks setup"
    fi
    
    # Create development directories
    log_info "Creating development directories..."
    mkdir -p logs/{training,evaluation,experiments}
    mkdir -p outputs/{models,visualizations,reports}
    touch logs/.gitkeep outputs/.gitkeep
    log_success "Development directories created"
fi

# Setup data directories
log_header "Setting Up Data Structure"

log_info "Creating data directories..."
mkdir -p data/{raw,processed,splits}
mkdir -p models/{checkpoints,exports}
mkdir -p configs
mkdir -p logs/{training,evaluation}

# Create .gitkeep files for empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/splits/.gitkeep
touch models/checkpoints/.gitkeep
touch models/exports/.gitkeep
touch logs/training/.gitkeep
touch logs/evaluation/.gitkeep

log_success "Data directories created"

# Create basic configuration files
log_header "Creating Configuration Files"

# Create default training config
if [ ! -f "configs/default.yaml" ]; then
    log_info "Creating default training configuration..."
    cat > configs/default.yaml << 'EOF'
# oaSentinel Default Training Configuration
model:
  architecture: "yolov8m"
  pretrained: true
  num_classes: 1  # Human detection only

dataset:
  name: "crowdhuman"
  path: "data/processed/crowdhuman"
  train_split: 0.8
  val_split: 0.15
  test_split: 0.05
  image_size: 640
  batch_size: 16

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "AdamW"
  scheduler: "cosine"
  device: "auto"
  workers: 4
  patience: 10  # Early stopping
  
  # Augmentation
  augment:
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
    degrees: 0.0
    translate: 0.1
    scale: 0.5
    shear: 0.0
    perspective: 0.0
    flipud: 0.0
    fliplr: 0.5
    mosaic: 1.0
    mixup: 0.0

evaluation:
  iou_threshold: 0.5
  confidence_threshold: 0.25
  metrics: ["mAP@0.5", "mAP@0.5:0.95", "precision", "recall"]

export:
  formats: ["onnx", "coreml"]
  optimize: true
  quantize: "int8"
  simplify: true

experiment:
  name: "crowdhuman_baseline"
  project: "oaSentinel"
  notes: "Baseline training on CrowdHuman dataset"
  tags: ["baseline", "crowdhuman", "yolov8m"]
EOF
    log_success "Default configuration created"
fi

# Create environment file template
if [ ! -f ".env.example" ]; then
    log_info "Creating environment template..."
    cat > .env.example << 'EOF'
# oaSentinel Environment Configuration

# Weights & Biases (experiment tracking)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=oaSentinel
WANDB_ENTITY=orangead

# Dataset paths
CROWDHUMAN_PATH=data/raw/crowdhuman
CUSTOM_DATASET_PATH=data/raw/custom

# Model storage
MODEL_REGISTRY_URL=s3://orangead-models/oaSentinel
CHECKPOINT_DIR=models/checkpoints
EXPORT_DIR=models/exports

# Training configuration
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4

# Deployment
DEPLOYMENT_REGISTRY=orangead-registry.azurecr.io
MODEL_VERSION=latest
EOF
    log_success "Environment template created"
fi

# Setup basic .gitignore
if [ ! -f ".gitignore" ]; then
    log_info "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Data and models (large files)
data/raw/
models/checkpoints/*.pt
models/checkpoints/*.pth
models/exports/*.onnx
models/exports/*.mlpackage
models/exports/*.engine

# Logs and outputs
logs/*.log
outputs/
wandb/
runs/
mlruns/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Coverage and testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json
EOF
    log_success ".gitignore created"
fi

# Verify installation
log_header "Verifying Installation"

log_info "Checking installed packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

if [ "$GPU_SUPPORT" = true ]; then
    log_info "Checking GPU support..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        python -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
    fi
fi

log_success "Installation verification complete"

# Display next steps
log_header "Setup Complete!"

echo -e "${GREEN}oaSentinel is ready for development!${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     ${BLUE}source $VENV_NAME/bin/activate${NC}"
echo ""
echo "  2. Copy and configure environment variables:"
echo "     ${BLUE}cp .env.example .env${NC}"
echo "     ${BLUE}# Edit .env with your API keys and preferences${NC}"
echo ""
echo "  3. Download the CrowdHuman dataset:"
echo "     ${BLUE}./scripts/download_data.sh${NC}"
echo ""
echo "  4. Start training a baseline model:"
echo "     ${BLUE}./scripts/train.sh --config configs/default.yaml${NC}"
echo ""
if [ "$DEV_MODE" = true ]; then
    echo "Development environment setup:"
    echo "  - Pre-commit hooks: ${GREEN}✓${NC}"
    echo "  - Development directories: ${GREEN}✓${NC}"
    echo "  - Code quality tools: ${GREEN}✓${NC}"
    echo ""
fi

echo -e "${BLUE}For more information, see the README.md file.${NC}"
echo -e "${BLUE}Happy training! 🚀${NC}"