# Trainer import is optional - only needed for training, not for inference
try:
    from .trainer import Trainer
except ImportError:
    # Allow inference without training dependencies
    Trainer = None
