#!/bin/bash

# oaSentinel Model Evaluation Script
# Comprehensive evaluation of trained YOLO models
# Usage: ./scripts/evaluate.sh --model path [--dataset name] [--split test]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH=""
DATASET="crowdhuman"
SPLIT="val"
OUTPUT_DIR=""
SAVE_PLOTS=true
DEVICE="auto"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${BLUE}===== $1 =====${NC}"; }

show_usage() {
    echo "oaSentinel Model Evaluation Script"
    echo ""
    echo "Usage: $0 --model MODEL_PATH [options]"
    echo ""
    echo "Required:"
    echo "  --model PATH      Path to trained model (.pt file)"
    echo ""
    echo "Options:"
    echo "  --dataset NAME    Dataset to evaluate on (default: crowdhuman)"
    echo "  --split SPLIT     Dataset split (train/val/test, default: val)"
    echo "  --output DIR      Output directory for results"
    echo "  --device DEVICE   Evaluation device (auto/cpu/gpu/0/1/...)"
    echo "  --no-plots        Skip generating evaluation plots"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model models/checkpoints/best.pt"
    echo "  $0 --model best.pt --split test --output results/"
    echo "  $0 --model best.pt --dataset custom --split val"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-plots)
            SAVE_PLOTS=false
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

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    log_error "Model path is required. Use --model to specify."
    show_usage
    exit 1
fi

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

# Validate model file
if [ ! -f "$MODEL_PATH" ]; then
    log_error "Model file not found: $MODEL_PATH"
    exit 1
fi

# Set default output directory
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH" .pt)
    OUTPUT_DIR="outputs/evaluation/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUTPUT_DIR"

log_header "oaSentinel Model Evaluation"
log_info "Model: $MODEL_PATH"
log_info "Dataset: $DATASET"
log_info "Split: $SPLIT"
log_info "Output directory: $OUTPUT_DIR"
log_info "Device: $DEVICE"

# Run evaluation with Python
log_info "Starting model evaluation..."

python3 << 'EOF'
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import cv2
from tqdm import tqdm

def find_dataset_yaml(dataset_name, split):
    """Find the dataset YAML file"""
    candidates = [
        f"data/processed/{dataset_name}/{dataset_name}.yaml",
        f"data/processed/{dataset_name}/dataset.yaml", 
        f"data/splits/{dataset_name}/dataset.yaml"
    ]
    
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    
    return None

def load_dataset_info(dataset_yaml):
    """Load dataset configuration"""
    with open(dataset_yaml, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model_path, dataset_yaml, split, device, output_dir, save_plots):
    """Comprehensive model evaluation"""
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"Using dataset: {dataset_yaml}")
    dataset_info = load_dataset_info(dataset_yaml)
    
    # Get dataset paths
    dataset_root = Path(dataset_info['path'])
    if split == 'val':
        split_path = dataset_root / dataset_info['val']
    elif split == 'train':
        split_path = dataset_root / dataset_info['train']
    elif split == 'test':
        if 'test' not in dataset_info:
            print("ERROR: Test split not available in dataset")
            return None
        split_path = dataset_root / dataset_info['test']
    else:
        print(f"ERROR: Unknown split: {split}")
        return None
    
    print(f"Evaluating on split: {split}")
    print(f"Images path: {split_path}")
    
    # Check if split exists
    if not split_path.exists():
        print(f"ERROR: Split path does not exist: {split_path}")
        return None
    
    # Run validation
    print("\nRunning model validation...")
    results = model.val(
        data=dataset_yaml,
        split=split,
        device=device,
        save_json=True,
        save_hybrid=True,
        conf=0.001,  # Low confidence for complete evaluation
        iou=0.6,
        max_det=300,
        plots=save_plots,
        project=output_dir,
        name='evaluation'
    )
    
    # Extract metrics
    metrics = {
        'model_path': str(model_path),
        'dataset': dataset_yaml,
        'split': split,
        'evaluation_date': datetime.now().isoformat(),
        'device': str(device),
        'results': {}
    }
    
    if hasattr(results, 'results_dict'):
        metrics['results'] = results.results_dict
    
    # Additional metrics calculation
    if hasattr(results, 'box'):
        box_metrics = results.box
        metrics['detailed_metrics'] = {
            'mAP50': float(box_metrics.map50) if hasattr(box_metrics, 'map50') else None,
            'mAP50_95': float(box_metrics.map) if hasattr(box_metrics, 'map') else None,
            'precision': float(box_metrics.mp) if hasattr(box_metrics, 'mp') else None,
            'recall': float(box_metrics.mr) if hasattr(box_metrics, 'mr') else None,
            'f1_score': float(box_metrics.f1) if hasattr(box_metrics, 'f1') else None,
        }
        
        # Per-class metrics if available
        if hasattr(box_metrics, 'ap_class_index') and hasattr(box_metrics, 'ap'):
            class_names = dataset_info.get('names', ['person'])
            per_class_metrics = {}
            
            for i, class_idx in enumerate(box_metrics.ap_class_index):
                class_name = class_names[int(class_idx)] if int(class_idx) < len(class_names) else f'class_{class_idx}'
                per_class_metrics[class_name] = {
                    'AP50': float(box_metrics.ap50[i]) if len(box_metrics.ap50) > i else None,
                    'AP50_95': float(box_metrics.ap[i]) if len(box_metrics.ap) > i else None,
                }
            
            metrics['per_class_metrics'] = per_class_metrics
    
    # Save detailed metrics
    output_path = Path(output_dir)
    metrics_file = output_path / 'evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nEvaluation results saved to: {metrics_file}")
    
    # Generate summary report
    generate_summary_report(metrics, output_path / 'evaluation_summary.txt')
    
    # Generate additional visualizations if requested
    if save_plots:
        generate_additional_plots(model, dataset_yaml, split, output_path)
    
    return metrics

def generate_summary_report(metrics, output_file):
    """Generate human-readable evaluation summary"""
    
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("oaSentinel Model Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic info
        f.write(f"Model: {metrics['model_path']}\n")
        f.write(f"Dataset: {metrics['dataset']}\n")
        f.write(f"Split: {metrics['split']}\n")
        f.write(f"Evaluation Date: {metrics['evaluation_date']}\n")
        f.write(f"Device: {metrics['device']}\n\n")
        
        # Key metrics
        if 'detailed_metrics' in metrics:
            dm = metrics['detailed_metrics']
            f.write("Key Performance Metrics:\n")
            f.write("-" * 30 + "\n")
            
            if dm.get('mAP50') is not None:
                f.write(f"mAP@0.5:      {dm['mAP50']:.4f}\n")
            if dm.get('mAP50_95') is not None:
                f.write(f"mAP@0.5:0.95: {dm['mAP50_95']:.4f}\n")
            if dm.get('precision') is not None:
                f.write(f"Precision:    {dm['precision']:.4f}\n")
            if dm.get('recall') is not None:
                f.write(f"Recall:       {dm['recall']:.4f}\n")
            if dm.get('f1_score') is not None:
                f.write(f"F1 Score:     {dm['f1_score']:.4f}\n")
        
        # Per-class metrics
        if 'per_class_metrics' in metrics:
            f.write("\nPer-Class Metrics:\n")
            f.write("-" * 30 + "\n")
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                f.write(f"\n{class_name}:\n")
                if class_metrics.get('AP50') is not None:
                    f.write(f"  AP@0.5:      {class_metrics['AP50']:.4f}\n")
                if class_metrics.get('AP50_95') is not None:
                    f.write(f"  AP@0.5:0.95: {class_metrics['AP50_95']:.4f}\n")
        
        # Performance interpretation
        f.write("\nPerformance Interpretation:\n")
        f.write("-" * 30 + "\n")
        
        if 'detailed_metrics' in metrics and metrics['detailed_metrics'].get('mAP50'):
            map50 = metrics['detailed_metrics']['mAP50']
            if map50 >= 0.9:
                interpretation = "Excellent"
            elif map50 >= 0.8:
                interpretation = "Very Good"
            elif map50 >= 0.7:
                interpretation = "Good"
            elif map50 >= 0.6:
                interpretation = "Fair"
            else:
                interpretation = "Needs Improvement"
            
            f.write(f"Overall Performance: {interpretation} (mAP@0.5: {map50:.3f})\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Summary report saved to: {output_file}")

def generate_additional_plots(model, dataset_yaml, split, output_dir):
    """Generate additional evaluation plots"""
    
    print("Generating additional visualizations...")
    
    # Create plots directory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # TODO: Add more custom visualizations here
    # - Confidence distribution plots
    # - Size distribution analysis
    # - Error analysis plots
    # - Comparison with baseline models
    
    print(f"Additional plots saved to: {plots_dir}")

def main():
    # Get parameters from environment
    model_path = os.environ.get('MODEL_PATH', '')
    dataset = os.environ.get('DATASET', 'crowdhuman')
    split = os.environ.get('SPLIT', 'val')
    device = os.environ.get('DEVICE', 'auto')
    output_dir = os.environ.get('OUTPUT_DIR', 'outputs/evaluation')
    save_plots = os.environ.get('SAVE_PLOTS', 'true').lower() == 'true'
    
    # Find dataset YAML
    dataset_yaml = find_dataset_yaml(dataset, split)
    if not dataset_yaml:
        print(f"ERROR: Could not find dataset YAML for {dataset}")
        print("Make sure you have processed the dataset with ./scripts/process_data.sh")
        sys.exit(1)
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 0
            print(f"CUDA available: Using GPU {device}")
        else:
            device = 'cpu'
            print("CUDA not available: Using CPU")
    
    # Run evaluation
    try:
        results = evaluate_model(
            model_path, dataset_yaml, split, device, output_dir, save_plots
        )
        
        if results:
            print("\n" + "="*50)
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*50)
            
            # Print key metrics
            if 'detailed_metrics' in results:
                dm = results['detailed_metrics']
                print(f"mAP@0.5: {dm.get('mAP50', 'N/A')}")
                print(f"mAP@0.5:0.95: {dm.get('mAP50_95', 'N/A')}")
                print(f"Precision: {dm.get('precision', 'N/A')}")
                print(f"Recall: {dm.get('recall', 'N/A')}")
            
            print(f"\nDetailed results saved to: {output_dir}")
        else:
            print("Evaluation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Set environment variables for Python script
export MODEL_PATH="$MODEL_PATH"
export DATASET="$DATASET"
export SPLIT="$SPLIT"
export DEVICE="$DEVICE"
export OUTPUT_DIR="$OUTPUT_DIR"
export SAVE_PLOTS="$SAVE_PLOTS"

# Check exit status
if [ $? -eq 0 ]; then
    log_header "Evaluation Completed Successfully!"
    log_success "Results saved to: $OUTPUT_DIR"
    log_info "Summary report: $OUTPUT_DIR/evaluation_summary.txt"
    log_info "Detailed metrics: $OUTPUT_DIR/evaluation_metrics.json"
    
    if [ "$SAVE_PLOTS" = true ]; then
        log_info "Visualization plots: $OUTPUT_DIR/plots/"
    fi
    
    echo ""
    log_info "Next step: ./scripts/export.sh --model $MODEL_PATH"
else
    log_error "Evaluation failed. Check the logs above for details."
    exit 1
fi