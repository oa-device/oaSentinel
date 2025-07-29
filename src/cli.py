#!/usr/bin/env python3
"""
oaSentinel CLI - Command Line Interface for oaSentinel AI model operations
"""

import typer
from pathlib import Path
from typing import Optional, List
from rich import print
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="oa-sentinel",
    help="🎯 oaSentinel: Custom AI models for OrangeAd human detection",
    rich_markup_mode="rich"
)

console = Console()

@app.command()
def info():
    """Show oaSentinel project information"""
    print("\n[bold blue]🎯 oaSentinel - OrangeAd Custom AI Models[/bold blue]")
    print("━" * 50)
    print("Advanced human detection and tracking optimization")
    print("Built on proven ML infrastructure from oaTracker")
    print()
    
    # Project status
    project_root = Path(__file__).parent.parent
    
    table = Table(title="Project Status", show_header=False)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    # Check for key directories and files
    checks = [
        ("Data directory", project_root / "data"),
        ("Models directory", project_root / "models"),
        ("Configuration", project_root / "configs"),
        ("Scripts", project_root / "scripts"),
        ("Virtual environment", project_root / ".venv"),
    ]
    
    for name, path in checks:
        status = "✅ Ready" if path.exists() else "❌ Missing"
        table.add_row(name, status)
    
    console.print(table)
    print()

@app.command()
def setup(
    clean: bool = typer.Option(False, "--clean", help="Clean installation"),
    gpu: bool = typer.Option(False, "--gpu", help="Install GPU support"),
    dev: bool = typer.Option(False, "--dev", help="Development setup")
):
    """Setup oaSentinel environment"""
    import subprocess
    import sys
    
    project_root = Path(__file__).parent.parent
    setup_script = project_root / "setup.sh"
    
    if not setup_script.exists():
        print("[red]❌ Setup script not found[/red]")
        raise typer.Exit(1)
    
    args = [str(setup_script)]
    if clean:
        args.append("--clean")
    if gpu:
        args.append("--gpu") 
    if dev:
        args.append("--dev")
    
    print(f"[blue]🔧 Running setup: {' '.join(args)}[/blue]")
    result = subprocess.run(args, cwd=project_root)
    
    if result.returncode != 0:
        print("[red]❌ Setup failed[/red]")
        raise typer.Exit(result.returncode)

@app.command()
def download(
    dataset: str = typer.Option("crowdhuman", "--dataset", help="Dataset to download"),
    force: bool = typer.Option(False, "--force", help="Force re-download")
):
    """Download training datasets"""
    import subprocess
    
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "download_data.sh"
    
    args = [str(script), "--dataset", dataset]
    if force:
        args.append("--force")
    
    print(f"[blue]📥 Downloading {dataset} dataset...[/blue]")
    result = subprocess.run(args, cwd=project_root)
    
    if result.returncode != 0:
        print("[red]❌ Download failed[/red]")
        raise typer.Exit(result.returncode)

@app.command()
def process(
    dataset: str = typer.Option("crowdhuman", "--dataset", help="Dataset to process"),
    splits: str = typer.Option("0.8,0.15,0.05", "--splits", help="Train/val/test splits")
):
    """Process datasets for training"""
    import subprocess
    
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "process_data.sh"
    
    args = [str(script), "--dataset", dataset, "--splits", splits]
    
    print(f"[blue]⚙️ Processing {dataset} dataset...[/blue]")
    result = subprocess.run(args, cwd=project_root)
    
    if result.returncode != 0:
        print("[red]❌ Processing failed[/red]")
        raise typer.Exit(result.returncode)

@app.command()
def train(
    config: Optional[Path] = typer.Option(None, "--config", help="Training configuration file"),
    model: Optional[str] = typer.Option(None, "--model", help="Model architecture"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of epochs"),
    device: str = typer.Option("auto", "--device", help="Training device"),
    wandb: bool = typer.Option(False, "--wandb", help="Enable W&B tracking"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume from checkpoint")
):
    """Train YOLO models"""
    import subprocess
    
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "train.sh"
    
    args = [str(script)]
    
    if config:
        args.extend(["--config", str(config)])
    if model:
        args.extend(["--model", model])
    if epochs:
        args.extend(["--epochs", str(epochs)])
    if device != "auto":
        args.extend(["--device", device])
    if wandb:
        args.append("--wandb")
    if resume:
        args.extend(["--resume", str(resume)])
    
    print("[blue]🚀 Starting model training...[/blue]")
    result = subprocess.run(args, cwd=project_root)
    
    if result.returncode != 0:
        print("[red]❌ Training failed[/red]")
        raise typer.Exit(result.returncode)

@app.command()
def evaluate(
    model: Path = typer.Argument(..., help="Path to trained model"),
    dataset: str = typer.Option("crowdhuman", "--dataset", help="Dataset to evaluate on"),
    split: str = typer.Option("val", "--split", help="Dataset split"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output directory"),
    device: str = typer.Option("auto", "--device", help="Evaluation device"),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plots")
):
    """Evaluate trained models"""
    import subprocess
    
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "evaluate.sh"
    
    args = [str(script), "--model", str(model), "--dataset", dataset, "--split", split]
    
    if output:
        args.extend(["--output", str(output)])
    if device != "auto":
        args.extend(["--device", device])
    if no_plots:
        args.append("--no-plots")
    
    print(f"[blue]📊 Evaluating model: {model.name}[/blue]")
    result = subprocess.run(args, cwd=project_root)
    
    if result.returncode != 0:
        print("[red]❌ Evaluation failed[/red]")
        raise typer.Exit(result.returncode)

@app.command()
def export(
    model: Path = typer.Argument(..., help="Path to trained model"),
    formats: str = typer.Option("onnx,coreml", "--formats", help="Export formats"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output directory"),
    image_size: int = typer.Option(640, "--image-size", help="Input image size"),
    quantize: Optional[str] = typer.Option(None, "--quantize", help="Quantization type"),
    no_optimize: bool = typer.Option(False, "--no-optimize", help="Disable optimization")
):
    """Export models for deployment"""
    import subprocess
    
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "export.sh"
    
    args = [str(script), "--model", str(model), "--formats", formats]
    
    if output:
        args.extend(["--output", str(output)])
    if image_size != 640:
        args.extend(["--image-size", str(image_size)])
    if quantize:
        args.extend(["--quantize", quantize])
    if no_optimize:
        args.append("--no-optimize")
    
    print(f"[blue]📦 Exporting model: {model.name}[/blue]")
    result = subprocess.run(args, cwd=project_root)
    
    if result.returncode != 0:
        print("[red]❌ Export failed[/red]")
        raise typer.Exit(result.returncode)

@app.command()
def list_models(
    directory: Path = typer.Option(Path("models/checkpoints"), "--dir", help="Models directory")
):
    """List available trained models"""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / directory
    
    if not models_dir.exists():
        print(f"[red]❌ Models directory not found: {models_dir}[/red]")
        raise typer.Exit(1)
    
    # Find .pt files
    model_files = list(models_dir.rglob("*.pt"))
    
    if not model_files:
        print("[yellow]⚠️ No trained models found[/yellow]")
        return
    
    table = Table(title="Available Models")
    table.add_column("Model", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Modified", style="magenta")
    
    for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        modified = model_file.stat().st_mtime
        from datetime import datetime
        modified_str = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
        
        relative_path = model_file.relative_to(project_root)
        table.add_row(
            model_file.name,
            str(relative_path),
            f"{size_mb:.1f} MB",
            modified_str
        )
    
    console.print(table)

@app.command()
def status():
    """Show project status and recent activity"""
    project_root = Path(__file__).parent.parent
    
    print("\n[bold blue]📊 oaSentinel Project Status[/bold blue]")
    print("━" * 50)
    
    # Check data
    data_dir = project_root / "data"
    if data_dir.exists():
        raw_datasets = list((data_dir / "raw").iterdir()) if (data_dir / "raw").exists() else []
        processed_datasets = list((data_dir / "processed").iterdir()) if (data_dir / "processed").exists() else []
        
        print(f"📂 Raw datasets: {len(raw_datasets)}")
        print(f"⚙️ Processed datasets: {len(processed_datasets)}")
    
    # Check models
    models_dir = project_root / "models"
    if models_dir.exists():
        checkpoints = list((models_dir / "checkpoints").rglob("*.pt")) if (models_dir / "checkpoints").exists() else []
        exports = list((models_dir / "exports").iterdir()) if (models_dir / "exports").exists() else []
        
        print(f"🎯 Model checkpoints: {len(checkpoints)}")
        print(f"📦 Exported models: {len(exports)}")
    
    # Recent activity
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.rglob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            from datetime import datetime
            modified = datetime.fromtimestamp(latest_log.stat().st_mtime)
            print(f"📝 Latest activity: {modified.strftime('%Y-%m-%d %H:%M')}")
    
    print()

if __name__ == "__main__":
    app()