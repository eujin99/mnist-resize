# mnist-resize

## MNIST Resize using Pillow (LANCZOS)

This repository contains a simple notebook that resizes the original **MNIST 28×28 images to 16×16 resolution** using the Pillow library with the `LANCZOS` interpolation algorithm.

## Features
- Load MNIST dataset
- Resize all images from 28x28 → 16x16
- Uses Pillow `Image.LANCZOS` for high-quality downsampling

## How to Run
```bash
jupyter notebook mnist_resize.ipynb
```

## Output
- Resized 16x16 MNIST images
- Sample images displayed inside the notebook
