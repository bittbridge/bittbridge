"""
Model Auto-Discovery Utility

Automatically finds and loads student models from the student_models/ folder.
"""

import os
import importlib
import importlib.util
import bittensor as bt
from typing import Optional
from ..model_interface import PredictionModel
from .function_wrapper import FunctionBasedModel


def load_student_model() -> Optional[PredictionModel]:
    """
    Auto-discover and load student model from student_models/ folder.
    
    Looks for .py files (excluding helpers.py and __init__.py) and loads
    the first one found. Expects a predict() function in the module.
    The default model file is my_model.py which is ready to use.
    
    Returns:
        PredictionModel instance, or None if no model found
    
    Example:
        model = load_student_model()
        if model:
            prediction, interval = model.predict("2024-01-15T10:30:00+00:00")
    """
    # Get the student_models directory path
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    student_models_dir = os.path.join(current_dir, 'student_models')
    
    if not os.path.exists(student_models_dir):
        bt.logging.error(f"Student models directory not found: {student_models_dir}")
        return None
    
    # Find Python files (excluding helpers and __init__)
    # Note: my_model.py is the default model file that students can use directly
    excluded_files = ['__init__.py', 'helpers.py']
    model_files = []
    for file in os.listdir(student_models_dir):
        if file.endswith('.py') and file not in excluded_files:
            model_files.append(file)
    
    if not model_files:
        bt.logging.error(
            "No student model found in student_models/ folder.\n"
            "The default my_model.py should be present. If you deleted it, restore it from the repository."
        )
        return None
    
    # Load the first model found
    model_file = model_files[0]
    module_name = model_file[:-3]  # Remove .py extension
    
    bt.logging.info(f"Found student model: {model_file}")
    
    try:
        # Load the module using importlib.util to properly handle imports
        # Set up the module with proper package context for relative imports
        import sys
        
        # Get the file path
        file_path = os.path.join(student_models_dir, model_file)
        
        # Create module spec
        spec = importlib.util.spec_from_file_location(
            f'miner_model.student_models.{module_name}',
            file_path
        )
        
        if spec is None or spec.loader is None:
            bt.logging.error(f"Could not create spec for {model_file}")
            return None
        
        # Create module and set package attribute for relative imports
        module = importlib.util.module_from_spec(spec)
        module.__package__ = 'miner_model.student_models'
        module.__name__ = f'miner_model.student_models.{module_name}'
        
        # Add parent directory to sys.path if needed for absolute imports
        parent_dir = os.path.dirname(student_models_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Execute the module
        spec.loader.exec_module(module)
        
        # Check if predict function exists
        if not hasattr(module, 'predict'):
            bt.logging.error(
                f"Model file {model_file} does not have a predict() function.\n"
                "Please make sure you've filled in SECTION 3 of the template."
            )
            return None
        
        # Wrap the predict function
        model = FunctionBasedModel(module.predict)
        bt.logging.success(f"Successfully loaded model from {model_file}")
        
        return model
        
    except Exception as e:
        bt.logging.error(f"Failed to load model from {model_file}: {e}")
        import traceback
        bt.logging.debug(traceback.format_exc())
        return None


def list_available_models() -> list:
    """
    List all available student models.
    
    Returns:
        List of model file names (excluding helpers.py and __init__.py)
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    student_models_dir = os.path.join(current_dir, 'student_models')
    
    if not os.path.exists(student_models_dir):
        return []
    
    excluded_files = ['__init__.py', 'helpers.py']
    models = []
    for file in os.listdir(student_models_dir):
        if file.endswith('.py') and file not in excluded_files:
            models.append(file)
    
    return models

