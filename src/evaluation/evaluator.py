"""
ModelEvaluator - Comprehensive model evaluation for oaSentinel
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
from datetime import datetime
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Comprehensive evaluation system for oaSentinel models
    
    Provides detailed performance analysis, visualization generation,
    and deployment readiness assessment.
    """
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to trained model (.pt file)
        """
        self.model_path = Path(model_path)
        self.model = YOLO(str(model_path))
        self.results = {}
    
    def evaluate(self,
                dataset_yaml: Union[str, Path],
                split: str = 'val',
                device: str = 'auto',
                output_dir: Optional[Union[str, Path]] = None,
                save_plots: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation
        
        Args:
            dataset_yaml: Path to dataset configuration
            split: Dataset split to evaluate on
            device: Evaluation device
            output_dir: Output directory for results
            save_plots: Whether to generate visualization plots
            
        Returns:
            Comprehensive evaluation results
        """
        if output_dir is None:
            model_name = self.model_path.stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(f"outputs/evaluation/{model_name}_{timestamp}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluating model: {self.model_path}")
        print(f"Dataset: {dataset_yaml}")
        print(f"Split: {split}")
        print(f"Output: {output_dir}")
        
        # Run validation
        results = self.model.val(
            data=str(dataset_yaml),
            split=split,
            device=device,
            save_json=True,
            save_hybrid=True,
            conf=0.001,  # Low confidence for complete evaluation
            iou=0.6,
            max_det=300,
            plots=save_plots,
            project=str(output_dir),
            name='evaluation'
        )
        
        # Compile comprehensive metrics
        evaluation_results = self._compile_results(
            results, dataset_yaml, split, device, output_dir
        )
        
        # Save results
        self._save_results(evaluation_results, output_dir)
        
        # Generate additional visualizations
        if save_plots:
            self._generate_custom_plots(evaluation_results, output_dir)
        
        self.results = evaluation_results
        return evaluation_results
    
    def _compile_results(self,
                        validation_results,
                        dataset_yaml: Union[str, Path], 
                        split: str,
                        device: str,
                        output_dir: Path) -> Dict[str, Any]:
        """Compile comprehensive evaluation results"""
        
        results = {
            'model_info': {
                'path': str(self.model_path),
                'name': self.model_path.name,
                'size_mb': self.model_path.stat().st_size / (1024 * 1024)
            },
            'evaluation_info': {
                'dataset': str(dataset_yaml),
                'split': split,
                'device': str(device),
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(output_dir)
            },
            'metrics': {},
            'performance_analysis': {},
            'deployment_readiness': {}
        }
        
        # Extract core metrics
        if hasattr(validation_results, 'box'):
            box_metrics = validation_results.box
            results['metrics'] = {
                'mAP50': float(box_metrics.map50) if hasattr(box_metrics, 'map50') else None,
                'mAP50_95': float(box_metrics.map) if hasattr(box_metrics, 'map') else None,
                'precision': float(box_metrics.mp) if hasattr(box_metrics, 'mp') else None,
                'recall': float(box_metrics.mr) if hasattr(box_metrics, 'mr') else None,
                'f1_score': float(box_metrics.f1) if hasattr(box_metrics, 'f1') else None,
            }
        
        # Performance analysis
        results['performance_analysis'] = self._analyze_performance(results['metrics'])
        
        # Deployment readiness assessment
        results['deployment_readiness'] = self._assess_deployment_readiness(results['metrics'])
        
        return results
    
    def _analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model performance and provide insights"""
        
        analysis = {
            'overall_grade': 'Unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        if metrics.get('mAP50') is not None:
            map50 = metrics['mAP50']
            
            # Overall grade
            if map50 >= 0.9:
                analysis['overall_grade'] = 'Excellent'
                analysis['strengths'].append('Outstanding detection accuracy')
            elif map50 >= 0.8:
                analysis['overall_grade'] = 'Very Good'
                analysis['strengths'].append('High detection accuracy')
            elif map50 >= 0.7:
                analysis['overall_grade'] = 'Good'
                analysis['strengths'].append('Solid detection performance')
            elif map50 >= 0.6:
                analysis['overall_grade'] = 'Fair'
                analysis['weaknesses'].append('Moderate detection accuracy')
                analysis['recommendations'].append('Consider additional training or data augmentation')
            else:
                analysis['overall_grade'] = 'Needs Improvement'
                analysis['weaknesses'].append('Low detection accuracy')
                analysis['recommendations'].append('Review training data quality and model architecture')
        
        # Precision/Recall analysis
        precision = metrics.get('precision')
        recall = metrics.get('recall')
        
        if precision is not None and recall is not None:
            if precision > 0.9:
                analysis['strengths'].append('High precision - few false positives')
            elif precision < 0.7:
                analysis['weaknesses'].append('Low precision - many false positives')
                analysis['recommendations'].append('Increase confidence threshold or improve training')
            
            if recall > 0.9:
                analysis['strengths'].append('High recall - detects most objects')
            elif recall < 0.7:
                analysis['weaknesses'].append('Low recall - misses many objects')
                analysis['recommendations'].append('Lower confidence threshold or add more training data')
        
        return analysis
    
    def _assess_deployment_readiness(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model readiness for production deployment"""
        
        assessment = {
            'ready_for_production': False,
            'ready_for_testing': False,
            'platform_suitability': {},
            'required_actions': []
        }
        
        map50 = metrics.get('mAP50', 0)
        
        # Production readiness thresholds
        if map50 >= 0.85:
            assessment['ready_for_production'] = True
            assessment['ready_for_testing'] = True
        elif map50 >= 0.75:
            assessment['ready_for_testing'] = True
            assessment['required_actions'].append('Validate performance on real-world data')
        else:
            assessment['required_actions'].append('Improve model accuracy before deployment')
        
        # Platform-specific suitability
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        
        assessment['platform_suitability'] = {
            'mac_mini': {
                'suitable': map50 >= 0.8,
                'performance_tier': 'high_accuracy' if map50 >= 0.85 else 'standard',
                'notes': 'CoreML format recommended for optimization'
            },
            'orangepi_5': {
                'suitable': map50 >= 0.75 and model_size_mb <= 100,
                'performance_tier': 'edge_optimized' if model_size_mb <= 50 else 'standard',
                'notes': 'ONNX INT8 quantization recommended for performance'
            },
            'ubuntu_gpu': {
                'suitable': map50 >= 0.8,
                'performance_tier': 'high_performance',
                'notes': 'ONNX format with CUDA optimization'
            }
        }
        
        return assessment
    
    def _save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation results to files"""
        
        # JSON results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Human-readable summary
        self._generate_summary_report(results, output_dir / 'evaluation_summary.txt')
        
        print(f"Results saved to: {output_dir}")
    
    def _generate_summary_report(self, results: Dict[str, Any], output_file: Path):
        """Generate human-readable evaluation summary"""
        
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("oaSentinel Model Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Model info
            model_info = results['model_info']
            f.write(f"Model: {model_info['name']}\n")
            f.write(f"Size: {model_info['size_mb']:.1f} MB\n")
            f.write(f"Evaluated: {results['evaluation_info']['timestamp']}\n\n")
            
            # Key metrics
            metrics = results['metrics']
            f.write("Performance Metrics:\n")
            f.write("-" * 25 + "\n")
            for metric, value in metrics.items():
                if value is not None:
                    f.write(f"{metric:12}: {value:.4f}\n")
            
            # Performance analysis
            analysis = results['performance_analysis']
            f.write(f"\nOverall Grade: {analysis['overall_grade']}\n\n")
            
            if analysis['strengths']:
                f.write("Strengths:\n")
                for strength in analysis['strengths']:
                    f.write(f"  ✅ {strength}\n")
                f.write("\n")
            
            if analysis['weaknesses']:
                f.write("Areas for Improvement:\n")
                for weakness in analysis['weaknesses']:
                    f.write(f"  ⚠️  {weakness}\n")
                f.write("\n")
            
            if analysis['recommendations']:
                f.write("Recommendations:\n")
                for rec in analysis['recommendations']:
                    f.write(f"  💡 {rec}\n")
                f.write("\n")
            
            # Deployment readiness
            deployment = results['deployment_readiness']
            f.write("Deployment Assessment:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Production Ready: {'✅ Yes' if deployment['ready_for_production'] else '❌ No'}\n")
            f.write(f"Testing Ready: {'✅ Yes' if deployment['ready_for_testing'] else '❌ No'}\n\n")
            
            # Platform suitability
            f.write("Platform Suitability:\n")
            for platform, info in deployment['platform_suitability'].items():
                status = "✅" if info['suitable'] else "❌"
                f.write(f"  {status} {platform}: {info['performance_tier']}\n")
                f.write(f"     {info['notes']}\n")
            
            f.write("\n" + "=" * 60 + "\n")
    
    def _generate_custom_plots(self, results: Dict[str, Any], output_dir: Path):
        """Generate additional custom visualization plots"""
        
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Performance overview radar chart
        self._plot_performance_radar(results['metrics'], plots_dir / 'performance_radar.png')
        
        # Deployment readiness visualization
        self._plot_deployment_readiness(results['deployment_readiness'], plots_dir / 'deployment_readiness.png')
        
        print(f"Custom plots saved to: {plots_dir}")
    
    def _plot_performance_radar(self, metrics: Dict[str, Any], output_path: Path):
        """Generate radar chart of performance metrics"""
        import numpy as np
        
        # Extract metrics for radar chart
        metric_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
        metric_keys = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']
        
        values = []
        for key in metric_keys:
            val = metrics.get(key, 0)
            values.append(val if val is not None else 0)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        values = values + [values[0]]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label='Model Performance')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.grid(True)
        
        plt.title('oaSentinel Model Performance Radar', size=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_deployment_readiness(self, deployment: Dict[str, Any], output_path: Path):
        """Generate deployment readiness visualization"""
        
        platforms = list(deployment['platform_suitability'].keys())
        suitability = [1 if deployment['platform_suitability'][p]['suitable'] else 0 for p in platforms]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(platforms, suitability, color=['green' if x else 'red' for x in suitability])
        
        # Add performance tier labels
        for i, (platform, bar) in enumerate(zip(platforms, bars)):
            tier = deployment['platform_suitability'][platform]['performance_tier']
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   tier, ha='center', va='bottom', rotation=45)
        
        ax.set_ylabel('Deployment Suitable')
        ax.set_title('Platform Deployment Readiness')
        ax.set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(self, other_evaluators: List['ModelEvaluator']) -> Dict[str, Any]:
        """
        Compare this model with other evaluated models
        
        Args:
            other_evaluators: List of other ModelEvaluator instances
            
        Returns:
            Comparison results
        """
        # TODO: Implement model comparison functionality
        pass