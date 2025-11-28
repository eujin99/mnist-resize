# -*- coding: utf-8 -*-
"""MNIST resize to 16x16 using Pillow LANCZOS"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torch
import random

# 1. Load MNIST
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=None
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=None
)

print("Train 개수:", len(train_dataset))
print("Test 개수 :", len(test_dataset))


# 2. Resize function (LANCZOS)
def resize_mnist_dataset(dataset, size=(16, 16)):
    """
    dataset: torchvision.datasets.MNIST (transform=None, PIL Image 반환)
    size: (width, height)

    return:
        X: np.ndarray, shape (N, H, W), dtype uint8 (0~255)
        y: np.ndarray, shape (N,)
    """
    N = len(dataset)
    W, H = size

    X = np.zeros((N, H, W), dtype=np.uint8)
    y = np.zeros((N,), dtype=np.int64)

    for i in range(N):
        img_pil, label = dataset[i]
        img_16 = img_pil.resize((W, H), resample=Image.LANCZOS)

        X[i] = np.array(img_16, dtype=np.uint8)
        y[i] = int(label)

    return X, y



# 3. Apply resizing
X_train_16_np, y_train_np = resize_mnist_dataset(train_dataset)
X_test_16_np,  y_test_np  = resize_mnist_dataset(test_dataset)

print("Train 16x16 shape:", X_train_16_np.shape)
print("Test  16x16 shape:", X_test_16_np.shape)

# convert to torch tensors
X_train_16 = torch.from_numpy(X_train_16_np).unsqueeze(1)
y_train = torch.from_numpy(y_train_np)
X_test_16 = torch.from_numpy(X_test_16_np).unsqueeze(1)
y_test = torch.from_numpy(y_test_np)

print("Train tensor:", X_train_16.shape, y_train.shape)
print("Test  tensor:", X_test_16.shape, y_test.shape)
print("dtype (이미지):", X_train_16.dtype)


# 4. Show sample
idx = 0
orig_pil, orig_label = train_dataset[idx]
resized_16 = X_train_16[idx, 0].numpy()

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(orig_pil, cmap='gray')
plt.title(f"Original 28x28 (label={orig_label})")

plt.subplot(1,2,2)
plt.imshow(resized_16, cmap='gray')
plt.title("LANCZOS 16x16")
plt.show()



# 5. Save dataset
torch.save({"images": X_train_16, "labels": y_train}, "mnist_16_train_lanczos.pt")
torch.save({"images": X_test_16,  "labels": y_test},  "mnist_16_test_lanczos.pt")

print("save : mnist_16_train_lanczos.pt, mnist_16_test_lanczos.pt")


# 6. Reload + visualize random samples
data = torch.load("mnist_16_train_lanczos.pt")
X = data["images"]
y = data["labels"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("dtype:", X.dtype)



# final grid show
plt.figure(figsize=(15, 5))
for i in range(10):
    idx = random.randint(0, len(X)-1)
    img = X[idx][0].numpy()
    label = int(y[idx])

    plt.subplot(2,5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}", fontsize=12)

    plt.xticks(range(16))
    plt.yticks(range(16))
    plt.grid(color='red', linestyle='-', linewidth=0.3)
    plt.tick_params(axis='both', labelsize=7)
    plt.axis('on')

plt.tight_layout()
plt.show()
