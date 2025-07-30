"""
Metrics calculation for model evaluation
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json


class MetricsCalculator:
    """
    Calculate various metrics for object detection evaluation
    
    Supports standard detection metrics like mAP, precision, recall,
    and custom metrics for human detection evaluation.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator
        
        Args:
            class_names: List of class names for the model
        """
        self.class_names = class_names or ["person"]
        self.num_classes = len(self.class_names)
        
    def calculate_detection_metrics(self, 
                                  predictions: List[Dict[str, Any]], 
                                  ground_truth: List[Dict[str, Any]],
                                  iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate standard object detection metrics
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries  
            iou_threshold: IoU threshold for considering detection as correct
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mAP_50_95": 0.0,
            "total_predictions": len(predictions),
            "total_ground_truth": len(ground_truth)
        }
        
        # Placeholder implementation - in real scenario, this would
        # calculate actual mAP, precision, recall using IoU matching
        if len(predictions) > 0 and len(ground_truth) > 0:
            # This is a simplified placeholder calculation
            metrics["precision"] = min(len(ground_truth) / len(predictions), 1.0)
            metrics["recall"] = min(len(predictions) / len(ground_truth), 1.0)
            metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"] + 1e-7)
            metrics["mAP_50"] = (metrics["precision"] + metrics["recall"]) / 2
        
        return metrics
    
    def calculate_per_class_metrics(self, 
                                  results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics per class
        
        Args:
            results: Results dictionary from model evaluation
            
        Returns:
            Per-class metrics dictionary
        """
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "mAP_50": 0.0,
                "support": 0  # Number of ground truth instances
            }
        
        return per_class_metrics
    
    def calculate_confusion_matrix(self, 
                                 predictions: List[int], 
                                 ground_truth: List[int]) -> np.ndarray:
        """
        Calculate confusion matrix for classification results
        
        Args:
            predictions: Predicted class indices
            ground_truth: Ground truth class indices
            
        Returns:
            Confusion matrix as numpy array
        """
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        for pred, true in zip(predictions, ground_truth):
            if 0 <= pred < self.num_classes and 0 <= true < self.num_classes:
                matrix[true, pred] += 1
                
        return matrix
    
    def calculate_speed_metrics(self, 
                              inference_times: List[float],
                              preprocessing_times: List[float] = None,
                              postprocessing_times: List[float] = None) -> Dict[str, float]:
        """
        Calculate speed and performance metrics
        
        Args:
            inference_times: List of inference times in seconds
            preprocessing_times: List of preprocessing times in seconds
            postprocessing_times: List of postprocessing times in seconds
            
        Returns:
            Speed metrics dictionary
        """
        metrics = {
            "avg_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "min_inference_time": np.min(inference_times),
            "max_inference_time": np.max(inference_times),
            "fps": 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0.0,
            "total_samples": len(inference_times)
        }
        
        if preprocessing_times:
            metrics.update({
                "avg_preprocessing_time": np.mean(preprocessing_times),
                "std_preprocessing_time": np.std(preprocessing_times)
            })
            
        if postprocessing_times:
            metrics.update({
                "avg_postprocessing_time": np.mean(postprocessing_times),
                "std_postprocessing_time": np.std(postprocessing_times)
            })
            
        # Calculate total pipeline time if all components available
        if preprocessing_times and postprocessing_times:
            total_times = [
                pre + inf + post 
                for pre, inf, post in zip(preprocessing_times, inference_times, postprocessing_times)
            ]
            metrics.update({
                "avg_total_time": np.mean(total_times),
                "pipeline_fps": 1.0 / np.mean(total_times) if np.mean(total_times) > 0 else 0.0
            })
        
        return metrics
    
    def save_metrics(self, metrics: Dict[str, Any], output_path: Path):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Metrics dictionary to save
            output_path: Path to save metrics JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_metrics = convert_numpy(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
        print(f"Metrics saved to: {output_path}")
    
    def load_metrics(self, metrics_path: Path) -> Dict[str, Any]:
        """
        Load metrics from JSON file
        
        Args:
            metrics_path: Path to metrics JSON file
            
        Returns:
            Loaded metrics dictionary
        """
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        return metrics
    
    def print_metrics_summary(self, metrics: Dict[str, Any]):
        """
        Print formatted metrics summary
        
        Args:
            metrics: Metrics dictionary to print
        """
        print("\n" + "="*50)
        print("EVALUATION METRICS SUMMARY")
        print("="*50)
        
        # Detection metrics
        if "precision" in metrics:
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1-Score: {metrics['f1_score']:.3f}")
            
        if "mAP_50" in metrics:
            print(f"mAP@0.5: {metrics['mAP_50']:.3f}")
            
        if "mAP_50_95" in metrics:
            print(f"mAP@0.5:0.95: {metrics['mAP_50_95']:.3f}")
            
        # Speed metrics
        if "fps" in metrics:
            print(f"FPS: {metrics['fps']:.1f}")
            print(f"Avg Inference Time: {metrics.get('avg_inference_time', 0):.3f}s")
            
        # Dataset info
        if "total_predictions" in metrics:
            print(f"Total Predictions: {metrics['total_predictions']}")
            print(f"Total Ground Truth: {metrics['total_ground_truth']}")
            
        print("="*50)