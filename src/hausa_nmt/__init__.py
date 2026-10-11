"""Hausa-to-English neural machine translation coursework project."""
# This project uses PyTorch only. Prevent Transformers from probing an unrelated
# TensorFlow/Keras installation, which can be incompatible with Keras 3.
import os
os.environ["USE_TF"] = "0"

__version__ = "1.0.0"
