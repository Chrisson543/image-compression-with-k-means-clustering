import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from functions import *
from settings import *


img = mpimg.imread("tree.png")
original_shape = img.shape
X_img = np.reshape(img, (img.shape[0] * img.shape[1], 3))
new_img = np.zeros(X_img.shape)

n_iterations = 20

center_points = initialize_center_points(n_center_points, X_img)
X_clusters = get_nearest_cluster(X_img, center_points)

for i in range(n_iterations):
    X_clusters = get_nearest_cluster(X_img, center_points)
    center_points = calculate_average_center(X_img, X_clusters, center_points)

    print(f'iteration: {i}')

new_img = center_points[X_clusters]

save_with_matplotlib(new_img.reshape(original_shape))

show_plot(X_img, center_points)
show_images(X_img.reshape(original_shape), new_img.reshape(original_shape))