import cv2
import numpy as np
import pandas as pd
import os
import seaborn as sns
from IPython.display import Image, display

TRAIN_IMG_DIR = "../data/train_images/"
TRAIN_MASK_DIR = "../data/train_masks/"
VAL_IMG_DIR = "../data/val_images/"
VAL_MASK_DIR = "../data/val_masks/"


class Mask:
    def __init__(
        self,
        video_info,
        amount_of_frames=2,
        videos_path="../EchoNet-Dynamic/Videos/",
        images_files_array=[],
    ):
        self.video_info = video_info
        self.amount_of_frames = amount_of_frames
        self.path = videos_path
        self.images_files_array = images_files_array

    def set_images_files_array(self, images_files_array):
        self.images_files_array = images_files_array

    def generate_masks(
        self,
        action="show",
        mask_type="centroid",
        show_centroid=False,
        include_points=False,
        include_base_image=False,
        resize_touple=(0, 0),
    ):
        images_files_array = self.images_files_array
        path = self.path
        video_info = self.video_info
        amount_of_frames = self.amount_of_frames

        for file in images_files_array:
            path_file = os.path.join(path, file)
            cap = cv2.VideoCapture(path_file)
            frames = video_info[video_info.FileName == file]["Frame"].unique()

            for frame in frames[0:amount_of_frames]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                coor = video_info[
                    (video_info.FileName == file) & (video_info.Frame == frame)
                ]
                p1 = [(int(row["X1"]), int(row["Y1"])) for _, row in coor.iterrows()]
                p2 = [(int(row["X2"]), int(row["Y2"])) for _, row in coor.iterrows()]

                output_image = None

                if mask_type == "centroid":
                    output_image = self.centroid(
                        p1, p2, cap, show_centroid, include_base_image
                    )
                elif mask_type == "convex":
                    output_image = self.convex(p1, p2, cap)
                elif mask_type == "simple_sort":
                    output_image = self.simple_sort(p1, p2, cap)
                elif mask_type == "points":
                    output_image = self.points(p1, p2, cap)
                else:
                    print("There is no mask type called:", mask_type)
                    return

                if include_points:
                    output_image = self.add_points_to_mask(p1, p2, output_image)

                if not (resize_touple[0] == 0 or resize_touple[1] == 0):
                    output_image = cv2.resize(output_image, (500, 500))

                if action == "save":
                    name = os.path.join(
                        VAL_MASK_DIR, file[0:-4] + "_" + str(frame) + ".jpg"
                    )
                    cv2.imwrite(name, output_image)

                    _, frame_image = cap.read()
                    name_image = os.path.join(
                        VAL_IMG_DIR, file[0:-4] + "_" + str(frame) + ".jpg"
                    )
                    cv2.imwrite(name_image, frame_image)

                else:
                    display(Image(data=cv2.imencode(".jpg", output_image)[1].tobytes()))
        if action == "save":
            print("IMAGE SAVING DONE!")

    def points(self, p1, p2, cap, include_base_image):
        ret, img = cap.read()
        if ret:
            for point in p1:
                cv2.circle(img, point, 1, (0, 255, 0), -1)

            for point in p2:
                cv2.circle(img, point, 1, (0, 255, 0), -1)

        return img

    def centroid(self, p1, p2, cap, show_centroid, include_base_image=False):
        points = np.concatenate((p1, p2), axis=0)

        centroid = np.mean(points, axis=0)

        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

        sorted_points = points[np.argsort(angles)]

        poly = np.array(sorted_points, np.int32)

        ret, img = cap.read()
        if ret:
            if include_base_image:
                polygon = cv2.fillPoly(img, [poly], (255, 255, 255, 90))
                if show_centroid:
                    cv2.circle(
                        polygon,
                        (int(centroid[0]), int(centroid[1])),
                        1,
                        (0, 255, 0),
                        -1,
                    )
            else:
                img = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(img, [poly], 255)
        return img

    def convex(self, p1, p2, cap, include_base_image):
        points = np.concatenate((p1, p2), axis=0).tolist()
        sorted_points = cv2.convexHull(np.array(points, dtype=np.int32))
        sorted_points = np.vstack((sorted_points, [sorted_points[0]]))

        poly = np.array(sorted_points, np.int32)

        ret, img = cap.read()

        if ret:
            cv2.fillPoly(img, [poly], (255, 255, 255, 90))

        return img

    def simple_sort(self, p1, p2, cap, include_base_image):
        sorted_puntos1 = sorted(p1, key=lambda x: x[1])
        sorted_puntos2 = sorted(p2, key=lambda x: x[1])

        sorted_puntos2.reverse()

        poly = np.array(
            np.concatenate((sorted_puntos1, sorted_puntos2), axis=0).tolist(),
            np.int32,
        )

        ret, img = cap.read()
        if ret:
            cv2.fillPoly(img, [poly], (255, 255, 255, 90))

        return img

    def add_points_to_mask(self, p1, p2, img):
        for point in p1:
            cv2.circle(img, point, 1, (0, 255, 0), -1)

        for point in p2:
            cv2.circle(img, point, 1, (0, 255, 0), -1)

        return img
