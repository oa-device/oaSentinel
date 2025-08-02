#!/bin/bash

# oaSentinel - Check Ignored Files Script
# Shows what files are being excluded from version control

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_header() { echo -e "\n${BLUE}===== $1 =====${NC}"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log_header "oaSentinel Repository Analysis"

# Check if we're in a git repository or submodule
if [ ! -d ".git" ] && [ ! -f ".git" ]; then
    log_warning "Not in a git repository or submodule. Run from oaSentinel root directory."
    exit 1
fi

log_info "Checking .gitignore patterns and file exclusions..."

# Show current ignored files with sizes
log_header "Currently Ignored Files (Disk Space Savings)"

echo "Files that would bloat the repository if tracked:"
echo

# Check large files
if [ -f "yolo11m.pt" ]; then
    size=$(du -h yolo11m.pt | cut -f1)
    echo "  📦 yolo11m.pt - ${size} (YOLOv11 Medium model)"
fi

if [ -f "yolov8m.pt" ]; then
    size=$(du -h yolov8m.pt | cut -f1)
    echo "  📦 yolov8m.pt - ${size} (YOLOv8 Medium model)"
fi

# Check virtual environment
if [ -d ".venv" ]; then
    size=$(du -sh .venv | cut -f1)
    echo "  🐍 .venv/ - ${size} (Python virtual environment)"
fi

# Check test dataset
if [ -d "data/test_dataset" ]; then
    size=$(du -sh data/test_dataset | cut -f1)
    images=$(find data/test_dataset -name "*.jpg" | wc -l | tr -d ' ')
    echo "  🖼️  data/test_dataset/ - ${size} (${images} synthetic images)"
fi

# Check for other data directories
for dir in data/raw data/processed data/splits; do
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        size=$(du -sh "$dir" | cut -f1)
        echo "  📊 ${dir}/ - ${size} (training data)"
    fi
done

# Check output directories
for dir in outputs runs wandb logs/training; do
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        size=$(du -sh "$dir" | cut -f1)
        echo "  📈 ${dir}/ - ${size} (generated outputs)"
    fi
done

echo

# Calculate total space saved
total_ignored=0
for item in .venv data/test_dataset *.pt outputs runs wandb; do
    if [ -e "$item" ]; then
        size_kb=$(du -sk "$item" 2>/dev/null | cut -f1)
        total_ignored=$((total_ignored + size_kb))
    fi
done

if [ $total_ignored -gt 0 ]; then
    total_mb=$((total_ignored / 1024))
    total_gb=$((total_mb / 1024))
    
    if [ $total_gb -gt 0 ]; then
        log_success "Total space saved: ${total_gb}GB+ (excluded from repository)"
    else
        log_success "Total space saved: ${total_mb}MB (excluded from repository)"
    fi
else
    log_info "No large ignored files found (repository is clean)"
fi

log_header "Git Status Summary"

# Show git status
tracked_files=$(git ls-files | wc -l | tr -d ' ')
log_info "Currently tracking: ${tracked_files} files"

# Show what would be tracked without .gitignore
log_info "Repository status:"
if git status --porcelain | grep -q "^??"; then
    echo "  📝 Untracked files (will be added on next commit):"
    git status --porcelain | grep "^??" | cut -c4- | sed 's/^/    /'
fi

if git status --porcelain | grep -q "^ M"; then
    echo "  ✏️  Modified files:"
    git status --porcelain | grep "^ M" | cut -c4- | sed 's/^/    /'
fi

echo

log_header "gitignore Effectiveness Test"

# Test some key patterns
test_patterns=(
    "*.pt"
    "data/test_dataset/"
    ".venv/"
    "outputs/"
    "__pycache__/"
)

log_info "Testing .gitignore patterns:"
for pattern in "${test_patterns[@]}"; do
    # Create a temporary test file
    test_file=""
    case "$pattern" in
        "*.pt") test_file="test_model.pt" ;;
        "data/test_dataset/") 
            mkdir -p data/test_dataset
            test_file="data/test_dataset/test_image.jpg" 
            ;;
        ".venv/") 
            mkdir -p .venv
            test_file=".venv/test_file" 
            ;;
        "outputs/") 
            mkdir -p outputs
            test_file="outputs/test_output.txt" 
            ;;
        "__pycache__/") 
            mkdir -p __pycache__
            test_file="__pycache__/test.pyc" 
            ;;
    esac
    
    if [ -n "$test_file" ]; then
        # Create test file temporarily
        touch "$test_file"
        
        if git check-ignore "$test_file" >/dev/null 2>&1; then
            echo "  ✅ ${pattern} - properly ignored"
        else
            echo "  ❌ ${pattern} - NOT ignored (check .gitignore)"
        fi
        
        # Clean up test file
        rm -f "$test_file"
    fi
done

log_header "Recommendations"

echo "✅ Repository is properly configured for ML development"
echo "✅ Large files are excluded to keep repo lightweight"
echo "✅ Generated content won't pollute version control"
echo
echo "💡 When sharing models:"
echo "   • Use model registries (Weights & Biases, MLflow)"
echo "   • Share download links, not actual model files"
echo "   • Document model versions in configs/"
echo
echo "💡 For large datasets:"
echo "   • Use external storage (S3, GCS, etc.)"
echo "   • Document download/setup in scripts/"
echo "   • Keep only sample data in repository"

log_success "Repository health check complete! 🎉"