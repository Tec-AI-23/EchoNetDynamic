import cv2
import numpy as np
import pandas as pd
import os
import seaborn as sns
from IPython.display import Image, display
import FILE_PATHS

""" We really need to stop commenting things hehe
TRAIN_IMG_DIR = "../data/train_images/"
TRAIN_MASK_DIR = "../data/train_masks/"
VAL_IMG_DIR = "../data/val_images/"
VAL_MASK_DIR = "../data/val_masks/"
"""
VAL_MASK_DIR = FILE_PATHS.MASKS


class Mask:
    def __init__(
        self,
        file_info,
        amount_of_frames=2,
        files_path="../EchoNet-Dynamic/data/images/",
        images_files_array=[],
    ):
        self.file_info = file_info
        self.amount_of_frames = amount_of_frames
        self.path = files_path
        self.images_files_array = images_files_array

    def set_images_files_array(self, images_files_array):
        self.images_files_array = images_files_array

    def generate_masks(
        self,
        action="show",
        mask_type="centroid",
        landmarks="All",
        show_centroid=False,
        include_points=False,
        include_base_image=False,
        resize_touple=(0, 0),
    ):
        images_files_array = self.images_files_array
        path = self.path
        file_info = self.file_info
        amount_of_frames = self.amount_of_frames

        for file in images_files_array:
            path_file = os.path.join(path, file)
            info_file = file_info[file_info.File == file]
            cordinates = [(row["X"], row["Y"]) for _, row in info_file.iterrows()]

            output_image = None
            cap = cv2.imread(path_file)

            if mask_type == "centroid":
                output_image = self.centroid(
                    cordinates, show_centroid, cap, include_base_image
                )
            elif mask_type == "convex":
                output_image = self.convex(cordinates, cap, include_base_image)
            elif mask_type == "simple_sort":
                output_image = self.simple_sort(cordinates, cap, include_base_image)
            elif mask_type == "points":
                output_image = self.points(cordinates, cap)
            else:
                print("There is no mask type called:", mask_type)
                return

            if include_points:
                output_image = self.add_points_to_mask(cordinates, output_image)

            if not (resize_touple[0] == 0 or resize_touple[1] == 0):
                output_image = cv2.resize(output_image, (500, 500))

            if action == "save":
                name = os.path.join(VAL_MASK_DIR, file)
                cv2.imwrite(name, output_image)

            else:
                display(Image(data=cv2.imencode(".jpg", output_image)[1].tobytes()))
        if action == "save":
            print("IMAGE SAVING DONE!")

    def points(self, cordinates, cap):
        img = cap
        if img is not None:
            for point in cordinates:
                cv2.circle(img, point, 1, (0, 255, 0), -1)

        return img

    def centroid(self, coordinates, show_centroid, cap, include_base_image=False):
        centroid = np.mean(coordinates, axis=0)
        poly = np.array(coordinates, np.int32)
        img = None

        if cap is not None:
            img = cap
            if include_base_image:
                polygon = cv2.fillPoly(img, [poly], (255, 255, 255, 90))
                if show_centroid:
                    cv2.circle(
                        polygon,
                        (int(centroid[0]), int(centroid[1])),
                        1,
                        (0, 255, 0),
                        1,
                    )
            else:
                img = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(img, [poly], 255)

        return img

    def convex(self, cordinates, cap, include_base_image=False):
        points = cordinates.tolist()
        sorted_points = cv2.convexHull(np.array(points, dtype=np.int32))
        sorted_points = np.vstack((sorted_points, [sorted_points[0]]))
        poly = np.array(sorted_points, np.int32)
        img = cap

        if img is not None:
            if include_base_image:
                cv2.fillPoly(img, [poly], (255, 255, 255, 90))
            else:
                img = np.zeros(img.shape[:2], dtype=np.uint8)

        return img

    def simple_sort(self, cordinates, cap, include_base_image=False):
        sorted_puntos = sorted(cordinates, key=lambda x: x[1])

        img = cap
        if cap is not None:
            cv2.fillPoly(img, [sorted_puntos], (255, 255, 255, 90))

        return img

    def add_points_to_mask(self, cordinates, img):
        if img is not None:
            for point in cordinates:
                cv2.circle(img, point, 1, (0, 255, 0), -1)
        else:
            print("There is not image to use")
        return img
