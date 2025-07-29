"""
Tests for training pipeline
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from src.training import TrainingPipeline

class TestTrainingPipeline:
    """Test cases for TrainingPipeline"""
    
    def test_init_with_config_file(self, tmp_path):
        """Test initialization with configuration file"""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
        model:
          architecture: "yolov8n"
          pretrained: true
        dataset:
          name: "test"
          batch_size: 4
        training:
          epochs: 1
          learning_rate: 0.001
        """
        config_file.write_text(config_content)
        
        pipeline = TrainingPipeline(config_file)
        
        assert pipeline.config_path == config_file
        assert pipeline.config['model']['architecture'] == 'yolov8n'
        assert pipeline.config['training']['epochs'] == 1
    
    def test_config_loading_error(self, tmp_path):
        """Test error handling for missing config file"""
        config_file = tmp_path / "missing_config.yaml"
        
        with pytest.raises(FileNotFoundError):
            TrainingPipeline(config_file)
    
    @patch('src.training.trainer.YOLO')
    def test_initialize_model_pretrained(self, mock_yolo, tmp_path):
        """Test model initialization with pretrained weights"""
        config_file = tmp_path / "config.yaml"
        config_content = """
        model:
          architecture: "yolov8n"
          pretrained: true
        """
        config_file.write_text(config_content)
        
        pipeline = TrainingPipeline(config_file)
        pipeline.initialize_model()
        
        mock_yolo.assert_called_with("yolov8n.pt")
    
    @patch('src.training.trainer.YOLO')
    def test_initialize_model_from_scratch(self, mock_yolo, tmp_path):
        """Test model initialization from scratch"""
        config_file = tmp_path / "config.yaml" 
        config_content = """
        model:
          architecture: "yolov8n"
          pretrained: false
        """
        config_file.write_text(config_content)
        
        pipeline = TrainingPipeline(config_file)
        pipeline.initialize_model(pretrained=False)
        
        mock_yolo.assert_called_with("yolov8n.yaml")
    
    def test_find_dataset_yaml_success(self, tmp_path):
        """Test successful dataset YAML discovery"""
        config_file = tmp_path / "config.yaml"
        config_content = """
        model:
          architecture: "yolov8n"
        dataset:
          name: "test_dataset"
          path: "data/processed/test_dataset"
        """
        config_file.write_text(config_content)
        
        # Create mock dataset YAML
        dataset_dir = tmp_path / "data" / "processed" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        dataset_yaml = dataset_dir / "test_dataset.yaml"
        dataset_yaml.write_text("path: .")
        
        pipeline = TrainingPipeline(config_file)
        
        # Mock the current directory to be tmp_path
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            found_yaml = pipeline._find_dataset_yaml(pipeline.config['dataset'])
            assert Path(found_yaml).name == "test_dataset.yaml"
    
    def test_find_dataset_yaml_not_found(self, tmp_path):
        """Test dataset YAML not found error"""
        config_file = tmp_path / "config.yaml"
        config_content = """
        model:
          architecture: "yolov8n"
        dataset:
          name: "missing_dataset"
        """
        config_file.write_text(config_content)
        
        pipeline = TrainingPipeline(config_file)
        
        with pytest.raises(FileNotFoundError):
            pipeline._find_dataset_yaml(pipeline.config['dataset'])

@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration for testing"""
    config_file = tmp_path / "test_config.yaml"
    config_content = """
    model:
      architecture: "yolov8n"
      pretrained: true
      num_classes: 1
    
    dataset:
      name: "crowdhuman"
      path: "data/processed/crowdhuman"
      train_split: 0.8
      val_split: 0.15
      test_split: 0.05
      image_size: 640
      batch_size: 4
    
    training:
      epochs: 2
      learning_rate: 0.001
      optimizer: "AdamW"
      scheduler: "cosine"
      device: "cpu"
      workers: 1
      patience: 5
      
    export:
      formats: ["onnx"]
      optimize: true
      quantize: "int8"
    """
    config_file.write_text(config_content)
    return config_file