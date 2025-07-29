#!/bin/bash

# oaSentinel Model Export Script
# Export trained models to multiple formats for deployment
# Usage: ./scripts/export.sh --model path [--formats coreml,onnx] [--optimize]

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
FORMATS="onnx,coreml"
OPTIMIZE=true
QUANTIZE=""
OUTPUT_DIR=""
IMAGE_SIZE=640
DEVICE="cpu"  # Export on CPU for compatibility

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${BLUE}===== $1 =====${NC}"; }

show_usage() {
    echo "oaSentinel Model Export Script"
    echo ""
    echo "Usage: $0 --model MODEL_PATH [options]"
    echo ""
    echo "Required:"
    echo "  --model PATH        Path to trained model (.pt file)"
    echo ""
    echo "Options:"
    echo "  --formats LIST      Export formats (comma-separated)"
    echo "                      Available: onnx, coreml, torchscript, tflite"
    echo "  --output DIR        Output directory (default: models/exports)"
    echo "  --image-size SIZE   Input image size (default: 640)"
    echo "  --quantize TYPE     Quantization type (int8, fp16)"
    echo "  --no-optimize       Disable optimization"
    echo "  --device DEVICE     Export device (default: cpu)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model models/checkpoints/best.pt"
    echo "  $0 --model best.pt --formats onnx,coreml --quantize int8"
    echo "  $0 --model best.pt --formats onnx --output exports/v1.0/"
    echo ""
    echo "Export formats:"
    echo "  onnx         - Cross-platform ONNX format (recommended)"
    echo "  coreml       - Apple CoreML for macOS/iOS deployment"
    echo "  torchscript  - PyTorch TorchScript format"
    echo "  tflite       - TensorFlow Lite for mobile/edge"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --formats)
            FORMATS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --quantize)
            QUANTIZE="$2"
            shift 2
            ;;
        --no-optimize)
            OPTIMIZE=false
            shift
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
    OUTPUT_DIR="models/exports/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUTPUT_DIR"

log_header "oaSentinel Model Export"
log_info "Model: $MODEL_PATH"
log_info "Formats: $FORMATS"
log_info "Output directory: $OUTPUT_DIR"
log_info "Image size: $IMAGE_SIZE"
log_info "Optimization: $OPTIMIZE"
log_info "Quantization: ${QUANTIZE:-none}"
log_info "Device: $DEVICE"

# Convert formats string to array
IFS=',' read -ra FORMAT_ARRAY <<< "$FORMATS"

# Run export with Python
log_info "Starting model export..."

python3 << 'EOF'
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import platform

def check_format_availability():
    """Check which export formats are available"""
    available_formats = {
        'onnx': True,  # Should always be available
        'torchscript': True,  # Should always be available
        'coreml': platform.system() == 'Darwin',  # macOS only
        'tflite': False,  # Would need tensorflow
    }
    
    # Check for optional dependencies
    try:
        import coremltools
        available_formats['coreml'] = True
    except ImportError:
        available_formats['coreml'] = False
    
    try:
        import tensorflow as tf
        available_formats['tflite'] = True
    except ImportError:
        available_formats['tflite'] = False
    
    return available_formats

def export_model(model_path, formats, output_dir, image_size, optimize, quantize, device):
    """Export model to multiple formats"""
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Check format availability
    available_formats = check_format_availability()
    
    export_results = {
        'model_path': str(model_path),
        'export_date': datetime.now().isoformat(),
        'image_size': image_size,
        'optimization': optimize,
        'quantization': quantize,
        'device': device,
        'exports': {}
    }
    
    successful_exports = []
    failed_exports = []
    
    for fmt in formats:
        fmt = fmt.strip().lower()
        
        print(f"\n{'='*50}")
        print(f"Exporting to {fmt.upper()} format...")
        print(f"{'='*50}")
        
        if not available_formats.get(fmt, False):
            print(f"ERROR: {fmt} format not available on this system")
            failed_exports.append(fmt)
            continue
        
        try:
            export_args = {
                'format': fmt,
                'imgsz': image_size,
                'device': device,
                'optimize': optimize,
                'verbose': True
            }
            
            # Add format-specific arguments
            if fmt == 'onnx':
                export_args.update({
                    'dynamic': False,  # Static shapes for better compatibility
                    'simplify': optimize,
                    'opset': 12,  # ONNX opset version
                })
                
                if quantize == 'int8':
                    export_args['int8'] = True
                elif quantize == 'fp16':
                    export_args['half'] = True
            
            elif fmt == 'coreml':
                export_args.update({
                    'nms': True,  # Include NMS in CoreML model
                })
                
                if quantize == 'int8':
                    export_args['int8'] = True
                elif quantize == 'fp16':
                    export_args['half'] = True
            
            elif fmt == 'tflite':
                if quantize == 'int8':
                    export_args['int8'] = True
                elif quantize == 'fp16':
                    export_args['half'] = True
            
            # Perform export
            export_path = model.export(**export_args)
            
            # Move exported file to output directory
            exported_file = Path(export_path)
            if exported_file.exists():
                # Determine output filename
                if fmt == 'coreml':
                    # CoreML exports as .mlpackage directory
                    output_name = f"{Path(model_path).stem}.mlpackage"
                elif fmt == 'onnx':
                    output_name = f"{Path(model_path).stem}.onnx"
                elif fmt == 'torchscript':
                    output_name = f"{Path(model_path).stem}.torchscript"
                elif fmt == 'tflite':
                    output_name = f"{Path(model_path).stem}.tflite"
                else:
                    output_name = exported_file.name
                
                final_path = Path(output_dir) / output_name
                
                # Move/copy the exported file
                if exported_file.is_dir():
                    # For directories like .mlpackage
                    import shutil
                    if final_path.exists():
                        shutil.rmtree(final_path)
                    shutil.copytree(exported_file, final_path)
                else:
                    # For regular files
                    import shutil
                    shutil.copy2(exported_file, final_path)
                
                # Get file size
                if final_path.is_dir():
                    # Calculate directory size
                    size_bytes = sum(f.stat().st_size for f in final_path.rglob('*') if f.is_file())
                else:
                    size_bytes = final_path.stat().st_size
                
                size_mb = size_bytes / (1024 * 1024)
                
                export_info = {
                    'format': fmt,
                    'path': str(final_path),
                    'size_bytes': size_bytes,
                    'size_mb': round(size_mb, 2),
                    'success': True
                }
                
                export_results['exports'][fmt] = export_info
                successful_exports.append(fmt)
                
                print(f"✅ {fmt.upper()} export successful!")
                print(f"   Output: {final_path}")
                print(f"   Size: {size_mb:.2f} MB")
                
            else:
                raise FileNotFoundError(f"Exported file not found: {export_path}")
                
        except Exception as e:
            print(f"❌ {fmt.upper()} export failed: {e}")
            export_results['exports'][fmt] = {
                'format': fmt,
                'success': False,
                'error': str(e)
            }
            failed_exports.append(fmt)
    
    # Save export metadata
    metadata_file = Path(output_dir) / 'export_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(export_results, f, indent=2)
    
    # Generate deployment instructions
    generate_deployment_instructions(export_results, Path(output_dir))
    
    return successful_exports, failed_exports

def generate_deployment_instructions(export_results, output_dir):
    """Generate deployment instructions for exported models"""
    
    instructions_file = output_dir / 'DEPLOYMENT.md'
    
    with open(instructions_file, 'w') as f:
        f.write("# oaSentinel Model Deployment Guide\n\n")
        f.write(f"Generated: {export_results['export_date']}\n")
        f.write(f"Source Model: {export_results['model_path']}\n\n")
        
        # Per-format instructions
        for fmt, info in export_results['exports'].items():
            if not info['success']:
                continue
                
            f.write(f"## {fmt.upper()} Deployment\n\n")
            f.write(f"**File:** `{Path(info['path']).name}`\n")
            f.write(f"**Size:** {info['size_mb']} MB\n\n")
            
            if fmt == 'onnx':
                f.write("### Usage in oaTracker\n")
                f.write("```python\n")
                f.write("# Replace in oaTracker configuration\n")
                f.write(f"model_path = '{Path(info['path']).name}'\n")
                f.write("model_format = 'onnx'\n")
                f.write("```\n\n")
                f.write("### Deployment to OrangePi 5\n")
                f.write("```bash\n")
                f.write("# Copy to device\n")
                f.write(f"scp {Path(info['path']).name} pi@device:/opt/oatracker/models/\n")
                f.write("# Update oaTracker config\n")
                f.write("# Restart oaTracker service\n")
                f.write("```\n\n")
            
            elif fmt == 'coreml':
                f.write("### Usage in oaTracker (macOS)\n")
                f.write("```python\n")
                f.write("# macOS-specific oaTracker configuration\n")
                f.write(f"model_path = '{Path(info['path']).name}'\n")
                f.write("model_format = 'coreml'\n")
                f.write("```\n\n")
                f.write("### Deployment to Mac Mini\n")
                f.write("```bash\n")
                f.write("# Via Ansible\n")
                f.write("ansible-playbook -i inventory/spectra-prod.yml \\\n")
                f.write("  playbooks/universal.yml \\\n")
                f.write("  --tags tracker \\\n")
                f.write(f"  --extra-vars 'sentinel_model={Path(info['path']).name}'\n")
                f.write("```\n\n")
        
        # Integration notes
        f.write("## Integration Notes\n\n")
        f.write("1. **Model Validation**: Test the exported model with sample data before deployment\n")
        f.write("2. **Performance Testing**: Benchmark inference speed on target hardware\n")
        f.write("3. **Backup**: Keep the original .pt file for future exports or fine-tuning\n")
        f.write("4. **Version Control**: Tag model versions in your deployment system\n\n")
        
        # Performance expectations
        f.write("## Expected Performance\n\n")
        f.write("| Platform | Format | Est. Inference Time | Notes |\n")
        f.write("|----------|---------|-------------------|-------|\n")
        f.write("| Mac Mini | CoreML | 15-25ms | Optimized for Apple Silicon |\n")
        f.write("| OrangePi 5 | ONNX | 80-120ms | ARM CPU inference |\n")
        f.write("| Ubuntu GPU | ONNX | 8-15ms | CUDA acceleration |\n\n")
    
    print(f"Deployment instructions saved to: {instructions_file}")

def main():
    # Get parameters from environment
    model_path = os.environ.get('MODEL_PATH', '')
    formats_str = os.environ.get('FORMATS', 'onnx,coreml')
    output_dir = os.environ.get('OUTPUT_DIR', 'models/exports')
    image_size = int(os.environ.get('IMAGE_SIZE', '640'))
    optimize = os.environ.get('OPTIMIZE', 'true').lower() == 'true'
    quantize = os.environ.get('QUANTIZE', '')
    device = os.environ.get('DEVICE', 'cpu')
    
    # Parse formats
    formats = [f.strip() for f in formats_str.split(',')]
    
    print(f"Export configuration:")
    print(f"  Model: {model_path}")
    print(f"  Formats: {formats}")
    print(f"  Output: {output_dir}")
    print(f"  Image size: {image_size}")
    print(f"  Optimize: {optimize}")
    print(f"  Quantize: {quantize or 'none'}")
    print(f"  Device: {device}")
    
    # Check system capabilities
    print(f"\nSystem information:")
    print(f"  Platform: {platform.system()}")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    
    try:
        successful, failed = export_model(
            model_path, formats, output_dir, image_size, optimize, quantize, device
        )
        
        print("\n" + "="*60)
        print("EXPORT COMPLETED!")
        print("="*60)
        
        if successful:
            print(f"✅ Successfully exported: {', '.join(successful)}")
        
        if failed:
            print(f"❌ Failed exports: {', '.join(failed)}")
        
        print(f"\nExported models saved to: {output_dir}")
        print(f"Deployment guide: {output_dir}/DEPLOYMENT.md")
        
        if failed:
            sys.exit(1)
            
    except Exception as e:
        print(f"Export failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Set environment variables for Python script
export MODEL_PATH="$MODEL_PATH"
export FORMATS="$FORMATS"
export OUTPUT_DIR="$OUTPUT_DIR"
export IMAGE_SIZE="$IMAGE_SIZE"
export OPTIMIZE="$OPTIMIZE"
export QUANTIZE="$QUANTIZE"
export DEVICE="$DEVICE"

# Check exit status
if [ $? -eq 0 ]; then
    log_header "Export Completed Successfully!"
    log_success "Exported models saved to: $OUTPUT_DIR"
    log_info "Deployment guide: $OUTPUT_DIR/DEPLOYMENT.md"
    log_info "Export metadata: $OUTPUT_DIR/export_metadata.json"
    
    echo ""
    log_info "Ready for deployment to:"
    log_info "- macOS devices (CoreML): Via oaAnsible tracker role"
    log_info "- OrangePi devices (ONNX): Via oaTracker configuration"
    log_info "- Ubuntu GPU servers (ONNX): Direct oaTracker integration"
else
    log_error "Export failed. Check the logs above for details."
    exit 1
fi