# mnist-resize

## MNIST Resize using Pillow (LANCZOS)

This repository contains a simple notebook that resizes the original **MNIST 28×28 images to 16×16 resolution** using the Pillow library with the `LANCZOS` interpolation algorithm.

## Features
- Load MNIST dataset
- Resize all images from 28x28 → 16x16
- Uses Pillow `Image.LANCZOS` for high-quality downsampling

## Output
- Resized 16x16 MNIST images
- Generated dataset files:
- 
  - `mnist_16_train_lanczos.pt`
  - `mnist_16_test_lanczos.pt`

## MNIST 28×28 vs 16×16 (LANCZOS)
[<img width="1018" height="558" alt="image" src="https://github.com/user-attachments/assets/db557f51-1e6c-4ba5-b9d9-c25801842d08" />](https://chatgpt.com/backend-api/estuary/content?id=file_0000000053007208800ede007d3cf025&ts=490086&p=fs&cid=1&sig=083a2d10e6c08bccf3f5f01b945b13f232c1737a28c0e5d0418841cc557c607c&v=0)

## Sample of Resized 16×16 MNIST Images
<img width="1406" height="490" alt="image" src="https://github.com/user-attachments/assets/95fe5952-3ec0-416e-9a56-32967d653e36" />
