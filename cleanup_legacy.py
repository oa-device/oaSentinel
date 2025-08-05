#!/usr/bin/env python3
"""
oaSentinel Legacy Cleanup Script
Removes all duplicate, legacy, and fallback-based files
AGGRESSIVE CLEANUP - NO FALLBACKS
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def cleanup_legacy_files():
    """Remove all legacy files and directories"""
    
    print("🧹 oaSentinel Aggressive Legacy Cleanup")
    print("=" * 50)
    
    # Files to DELETE (legacy, duplicates, fallback-based)
    files_to_delete = [
        # Root directory clutter
        "train_yolo11m.py",           # Replaced by bin/oas-train
        "setup_shell_aliases.sh",     # Not needed in production
        "start_training_auto.sh",     # Replaced by bin/oas-train
        "crowdhuman.yaml",            # Moved to configs/
        
        # Legacy model files (move to models/ first)
        "yolo11m.pt",
        "yolov8m.pt",
    ]
    
    # Directories to DELETE (legacy scripts with fallbacks)
    dirs_to_delete = [
        "scripts",  # All scripts replaced by clean bin/ executables
    ]
    
    # Move model files to models/ before deletion
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_files = ["yolo11m.pt", "yolov8m.pt"]
    for model_file in model_files:
        source = PROJECT_ROOT / model_file
        target = models_dir / model_file
        
        if source.exists() and not target.exists():
            print(f"📦 Moving {model_file} to models/")
            shutil.move(str(source), str(target))
        elif source.exists():
            print(f"🗑️  Removing duplicate {model_file}")
            source.unlink()
    
    # Delete legacy files
    deleted_files = 0
    for file_path in files_to_delete:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"🗑️  Deleting {file_path}")
            if full_path.is_file():
                full_path.unlink()
            else:
                shutil.rmtree(full_path)
            deleted_files += 1
    
    # Delete legacy directories
    deleted_dirs = 0
    for dir_path in dirs_to_delete:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"🗑️  Deleting directory {dir_path}/")
            shutil.rmtree(full_path)
            deleted_dirs += 1
    
    # Clean up old src structure and move to new structure
    old_src = PROJECT_ROOT / "src"
    new_src = PROJECT_ROOT / "src" / "oasentinel"
    
    if old_src.exists() and not new_src.exists():
        print("🔄 Restructuring src/ directory...")
        
        # Create new structure
        new_src.mkdir(parents=True, exist_ok=True)
        
        # Move old modules (if they don't conflict with new ones)
        old_modules = ["training", "evaluation"]
        for module in old_modules:
            old_module_path = old_src / module
            new_module_path = new_src / module
            
            if old_module_path.exists() and not new_module_path.exists():
                print(f"📦 Moving {module}/ to new structure")
                shutil.move(str(old_module_path), str(new_module_path))
    
    print(f"\n✅ Cleanup completed!")
    print(f"   Deleted {deleted_files} files")
    print(f"   Deleted {deleted_dirs} directories")
    print(f"   Moved model files to models/")
    
    # Show new clean structure
    print(f"\n📁 New Clean Structure:")
    print("   bin/           - Executable entry points")
    print("   src/oasentinel/ - Core Python package")
    print("   configs/       - Configuration files")
    print("   models/        - Model files (.pt, .onnx, .coreml)")
    print("   data/          - Dataset storage")
    print("   tests/         - Test suite")


def verify_executables():
    """Verify new executables are in place"""
    
    print(f"\n🔍 Verifying New Executables:")
    
    executables = ["oas-train", "oas-process", "oas-export", "oas-download"]
    bin_dir = PROJECT_ROOT / "bin"
    
    all_good = True
    for exe in executables:
        exe_path = bin_dir / exe
        if exe_path.exists():
            print(f"   ✅ {exe}")
        else:
            print(f"   ❌ {exe} - MISSING")
            all_good = False
    
    if all_good:
        print(f"\n🎉 All executables ready!")
        print(f"Usage:")
        print(f"   bin/oas-download --output data/raw/crowdhuman")
        print(f"   bin/oas-process --input data/raw/crowdhuman --output data/processed")
        print(f"   bin/oas-train --data configs/crowdhuman.yaml --epochs 100 --device [0,1]")
        print(f"   bin/oas-export --model runs/detect/train/weights/best.pt --formats onnx coreml")
    else:
        print(f"\n❌ Some executables missing - check bin/ directory")
        sys.exit(1)


if __name__ == "__main__":
    cleanup_legacy_files()
    verify_executables()
    
    print(f"\n🚀 oaSentinel restructuring complete!")
    print(f"   Professional, fail-fast, no-fallback implementation ready")
    print(f"   All legacy scripts and fallback logic removed")
