import os
import torch
import cv2 as cv
import numpy as np
import pixel_expand
from skimage.filters import gaussian
import matplotlib.pyplot as plt


class HeatmapGeneration:
    def __init__(
        self,
        path_tensors=FILE_PATHS.HEATMAPS,
        path_masks=FILE_PATHS.MASKS,
        frame_info=None,
    ):
        self.path_toTensors = path_tensors
        self.path_masks = path_masks
        self.frame_info = frame_info

    def heatmap(
        self,
        distribution="Gaussian",
        kernel_shape=(3, 3),
        morphological_iterations=2,
        save_as_tensor=True,
        radius=10,
        show=False,
        num_lands=5
    ):
        files = self.frame_info["File"].unique()
        kernel = np.ones(kernel_shape, np.uint8)

        for file in files:
            path_mask = os.path.join(self.path_masks, file)
            path_tensor = ""
            coor_df = self.frame_info[self.frame_info.File == file]
            coors = [(row["X"], row["Y"]) for index, row in coor_df.iterrows()]
            optimal_landmarks = self.optimal_lands(coors, num_lands)
            
            contour = self.getContour(path_mask, morphological_iterations, kernel)
            result_img = np.zeros(contour.shape[:2] + (len(optimal_landmarks),), dtype=np.uint8)
            area_pixel = np.zeros(contour.shape[:2], dtype=np.uint8)

            fig, axs = None, None
            if show:
                fig, axs = plt.subplots(1, len(optimal_landmarks), figsize=(20, 20))
            

            result_heatmap = []
            for i in range(len(optimal_landmarks)):
                coor_x = optimal_landmarks[i][0]
                coor_y = optimal_landmarks[i][1]

                if distribution == "Euclidean":
                    path_tensor = os.path.join(self.path_toTensors, "euclidean")
                    heatmap = self.euclidean_distribution(
                        area_pixel, coor_x, coor_y, contour.shape[0], radius
                    )
                    heatmap = heatmap * contour
                    result_img[:, :, i] = heatmap
                    result_heatmap.append(heatmap)
                    if show:
                        axs[i].set_title('(' + str(coor_x) + ',' +str(coor_y) + ')')
                        axs[i].imshow(heatmap)
                        axs[i].axis("off")

                elif distribution == "Gaussian":
                    path_tensor = os.path.join(self.path_toTensors, "gaussian")
                    heatmap = self.gaussian_distribution(
                        area_pixel.shape, coor_x, coor_y, radius
                    )
                    heatmap = heatmap * contour
                    result_img[:, :, i] = heatmap
                    result_heatmap.append(heatmap)
                    if show:
                        axs[i].set_title('(' + str(coor_x) + ',' +str(coor_y) + ')')
                        axs[i].imshow(heatmap)
                        axs[i].axis("off")

                else:
                    raise ValueError(
                        distribution
                        + " is not a distribution type supported by this class"
                    )

            if save_as_tensor:
                img_as_tensor = torch.tensor(np.array(result_heatmap))
                name_tensor = file[:-4] + "pt"
                path_tensor = os.path.join(path_tensor, name_tensor)

                torch.save(img_as_tensor, path_tensor)

    def gaussian_distribution(self, shape, coor_x, coor_y, radius):
        pixel = np.zeros(shape, dtype=np.uint8)
        pixel[coor_y][coor_x] = 1.0
        variance = radius / 2
        gaussian_area = gaussian(pixel, sigma=[variance, variance])
        gaussian_scaled = self.scale_distribution(gaussian_area)
        return gaussian_scaled

    def euclidean_distribution(self, area_pixel, coor_x, coor_y, shape, radius):
        pixel_expanded = pixel_expand.expand_pixel(area_pixel, coor_y, coor_x, radius)
        x, y = np.meshgrid(np.arange(shape), np.arange(shape))
        distance = np.sqrt((x - coor_x) ** 2 + (y - coor_y) ** 2)
        distance = distance / distance.max()
        euclidean_matrix = 1 - distance
        euclidean_radius = euclidean_matrix * pixel_expanded
        euclidean_scaled = self.scale_distribution(euclidean_radius)
        return euclidean_scaled

    def scale_distribution(self, matrix):
        matrix_flatted = matrix.flatten()
        non_zeros = matrix_flatted[matrix_flatted != 0]
        min_value = non_zeros.min()
        max_value = matrix.max()
        result = (matrix - min_value) / (max_value - min_value)
        result = np.clip(result, 0, None)
        return result
    
    def getContour(self, path_mask, iterations, kernel):
        mask = cv.imread(path_mask)
        dilated_img = cv.dilate(mask, kernel, iterations=iterations)
        eroded_img = cv.erode(mask, kernel, iterations=iterations)
        contour = cv.bitwise_xor(dilated_img, eroded_img)
        _, contour = cv.threshold(contour, 200, 255, cv.THRESH_BINARY)
        contour = contour[:, :, 0] / 255.0
        return contour
    
    def optimal_lands(self, sorted_points, num_lands):
        angles = []
        landmark = []
        
        for i in range(len(sorted_points)):
            a = sorted_points[i-1]
            b = sorted_points[i]
            c = 0
            
            if i == len(sorted_points) - 1:
                c = sorted_points[0]
            else:
                c = sorted_points[i+1]
            
            ba = (a[0] - b[0], a[1] - b[1])
            bc = (c[0] - b[0], c[1] - b[1])
            
            result = np.clip(((ba[0] * bc[0]) + (ba[1] * bc[1])) / ((np.sqrt(ba[0]**2 + ba[1]**2)) * (np.sqrt(bc[0]**2 + bc[1]**2))), -1, 1)
            angle = np.arccos(result)
            
            angle = round(angle, 6)
            angles.append(angle)
            landmark.append(b)
            
        sorted_index = np.argsort(angles)
        optimal_lands = [sorted_points[i] for i in sorted_index]

        return optimal_lands[:num_lands]
