import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image


def show_images(img1, img2, title1="Original", title2="Compressed"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_plot(X_img, center_points):
    ax = plt.axes(projection='3d')

    ax.scatter(X_img[:10000, 0], X_img[:10000, 1], X_img[:10000, 2], c=X_img[:10000])
    ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2], c='black', marker='x', s=200, edgecolors='white')
    ax.set_title('3D Scatter Plot')
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.show()


def initialize_center_points(n_center_points, X):
    center_points = np.random.permutation(X)[:n_center_points]

    return center_points


def get_nearest_cluster(X, center_points):
    dists = np.linalg.norm(X[:, None] - center_points, axis=2)
    nearest_cluster = np.argmin(dists, axis=1)

    return nearest_cluster


def calculate_average_center(X, X_clusters, center_points):
    avg_points = np.zeros((center_points.shape[0], X.shape[1]))

    for i in range(len(center_points)):
        avg_points[i] = np.mean(X[X_clusters == i], axis=0)

    return avg_points


def save_with_pillow(img, filename="compressed.png"):
    """
    Save an image using Pillow.
    img: numpy array (H, W, 3) or (H, W, 4)
    filename: output file name
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    Image.fromarray(img).save(filename)
    print(f"Saved with Pillow → {filename}")


def save_with_matplotlib(img, filename="compressed.png"):
    """
    Save an image using Matplotlib.
    img: numpy array (H, W, 3) or (H, W, 4)
    filename: output file name
    """
    if img.dtype != np.uint8 and img.max() > 1:
        img = (img / 255.0).astype(float)  # scale to [0,1] if float in 0–255
    plt.imsave(filename, img)
    print(f"Saved with Matplotlib → {filename}")