# 🧩 Jigsaw Puzzle Solver using Deep Learning

Deep learning pipeline for reconstructing scrambled jigsaw puzzles using CNNs, Transformers, and U-Net.

## Overview

This project reconstructs a scrambled 3×3 jigsaw puzzle from shuffled image patches. The model learns both the correct spatial arrangement of patches and reconstructs missing border information using an end-to-end neural network architecture.

## Model Architecture

- Shared CNN Encoder
- Transformer with Self-Attention
- Soft Permutation Layer
- U-Net Refinement Network

## Dataset

- STL-10
- 80,000 training images
- 10,000 test images

## Results

- Test MAE: **0.0456**
- 68 training epochs
- 2.6 million parameters
- 75% lower MAE than the baseline (0.182 → 0.0456)

## Engineering Highlights

To reduce memory consumption while processing large datasets, I implemented a memory-efficient data generation pipeline using `numpy.memmap` and on-the-fly normalization.

This optimization significantly reduced RAM usage and the optimized notebook was later shared by the course instructor with the class as a reference implementation for students encountering memory limitations.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Jupyter Notebook
