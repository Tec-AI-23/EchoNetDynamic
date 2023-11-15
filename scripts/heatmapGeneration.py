import pixel_expand
import numpy as np
import cv2 as cv
import torch
import os


class heatmapGeneration():
    def __init__(self, path_tensors='../EchoNet-Dynamic/tensors/', path_masks='../EchoNet-Dynamic/masked/', frame_info=None):
        self.path_toTensors = path_tensors
        self.path_masks = path_masks
        self.frame_info = frame_info
    
    def euclidian_heatmap(self, kernel_shape=(3,3), morphological_iterations = 2, save_as_tensor=True, radius = 10):
        files = self.frame_info['File'].unique()
        kernel = np.ones(kernel_shape, np.uint8)
        
        for file in files:
            path_mask = os.path.join(self.path_masks, file)
            coor = self.frame_info[self.frame_info.File == file]
            puntos = [(row['X'], row['Y']) for index, row in coor.iterrows()]
            
            contour = self.getContour(path_mask, morphological_iterations, kernel)
            result_img = np.zeros(contour.shape[:2] + (len(puntos),), dtype=np.uint8)
            area_pixel = np.zeros(contour.shape[:2], dtype=np.uint8)
            
            for i in range(len(puntos)):
                coor_x = puntos[i][0]
                coor_y = puntos[i][0]
                area_pixel = pixel_expand.expand_pixel(area_pixel, coor_y, coor_x, radius)
                x, y = np.meshgrid(np.arange(contour.shape[0]), np.arange(contour.shape[0]))
                distance = np.sqrt((x - coor_x)**2 + (y - coor_y)**2)
                distance = distance / distance.max()
                euclidean_matrix = 1 - distance
                euclidean_radius = euclidean_matrix * area_pixel
                euclidean_scaled = self.scale_euclidean_matrix(euclidean_radius)
                heatmap_channel = euclidean_scaled * contour
                result_img[:,:,i] = heatmap_channel
            
            if save_as_tensor:
                img_as_tensor = torch.tensor(result_img)
                name_tensor = file[:-4] + 'pt'
                path_tensor = os.path.join(self.path_toTensors, name_tensor)
                torch.save(img_as_tensor, path_tensor)
            
    def scale_euclidean_matrix(self, matrix):
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
        contour = contour[:,:,0] / 255.0
        return contour