# oaSentinel

**⚠️ EXPERIMENTAL - NOT PRODUCTION READY**
*Abandoned proof-of-concept project*

Experimental model training system for custom human detection models using PyTorch and YOLOv11. This project was developed without real-world validation and is **not recommended for production use**.

## Status: ABANDONED

**🔴 DO NOT USE FOR PRODUCTION**

### Why Abandoned
- No real-world validation or business requirements
- Over-engineered solution without proven use case
- High resource requirements (GPU, 16GB+ RAM)
- Complexity exceeded actual needs

### Current State
- Infrastructure exists but is untested
- No trained models available
- High system requirements without proven benefits

## Installation (For Reference Only)

```bash
cd oaSentinel
uv sync

# Download CrowdHuman dataset
bin/oas-download --output data/raw/crowdhuman

# Process to YOLO format
bin/oas-process --input data/raw/crowdhuman --output data/processed
```

## Training (GPU Required)

```bash
# Train model (high GPU requirements)
bin/oas-train \
  --data configs/crowdhuman.yaml \
  --epochs 100 \
  --device [0,1]

# Export trained model
bin/oas-export \
  --model runs/detect/train/weights/best.pt \
  --formats onnx coreml
```

## Project Structure

```
oaSentinel/
├── bin/                    # Executable entry points
├── src/oasentinel/         # Core Python package
├── configs/                # Configuration files
├── models/                 # Model exports (.pt, .onnx, .coreml)
├── data/                   # Dataset storage (gitignored)
└── tests/                  # Test suite
```

## Requirements

- NVIDIA GPU with CUDA 11.8+
- Python 3.9+
- 16GB+ RAM, 8GB+ VRAM
- 50GB+ free storage

## Recommended Alternatives

- Use pre-trained models from official sources
- Leverage existing computer vision libraries
- Consider cloud-based ML services for custom training

---

**Note**: This is an abandoned experimental project. All code is untested and documentation may be misleading. Significant investment would be required to make production-ready.

**Last Updated**: October 2025
**Status**: Experimental - Abandoned