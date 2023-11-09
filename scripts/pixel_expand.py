import numpy as np
import matplotlib.pyplot as plt

def generate_pixel_circle(radius):
    grid_size = radius * 2 + 1
    z = np.zeros((grid_size, grid_size))
    n = len(z)
    m = len(z[0])

    center_x = n // 2
    center_y = m // 2

    I, J = np.meshgrid(np.arange(z.shape[0]), np.arange(z.shape[1]))

    # calculate distance of all points to centre
    dist = np.sqrt((I - center_x) ** 2 + (J - center_y) ** 2)

    # Assign value of 1 to those points where dist<radius:
    z[np.where(dist < radius)] = 1

    return z

def plot_matrix(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(matrix)
    ax.set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()

def expand_pixel(matrix, x, y, expansion_radius=5):
    kernel = generate_pixel_circle(expansion_radius)

    # Get coordinates to align both matrices
    center_x, center_y = x, y
    start_x = center_x - kernel.shape[0] // 2
    start_y = center_y - kernel.shape[1] // 2
    new_matrix = np.zeros_like(matrix, dtype=bool)
    height, width = matrix.shape

    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            px, py = start_x + i, start_y + j
            if 0 <= px < height and 0 <= py < width:
                new_matrix[px][py] = np.logical_or(matrix[px][py], kernel[i][j])

    return new_matrix