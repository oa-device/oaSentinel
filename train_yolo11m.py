#!/usr/bin/env python3
"""
YOLO11m Training Script for CrowdHuman Dataset
Implements the exact training command specified in requirements:

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="crowdhuman.yaml", epochs=100, imgsz=640, device=[0, 1])
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Train YOLO11m on CrowdHuman dataset')
    parser.add_argument('--data', default='crowdhuman.yaml', help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='[0,1]', help='Device(s) to use for training')
    parser.add_argument('--model', default='yolo11m.pt', help='Model checkpoint to load')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path('crowdhuman.yaml').exists():
        print("Error: crowdhuman.yaml not found. Run this from the oaSentinel root directory.")
        sys.exit(1)
    
    # Parse device argument
    device = args.device
    if device.startswith('[') and device.endswith(']'):
        # Parse device list like "[0,1]"
        try:
            device = eval(device)
        except:
            print(f"Warning: Could not parse device {args.device}, using auto")
            device = 'auto'
    
    print("YOLO11m Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {device}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
    print()
    
    # Load a model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)  # load a pretrained model (recommended for training)
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=args.data, 
        epochs=args.epochs, 
        imgsz=args.imgsz, 
        device=device
    )
    
    print("\nTraining completed!")
    print(f"Results: {results}")

if __name__ == '__main__':
    main()