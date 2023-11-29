import cv2
import os
import numpy as np
import pandas as pd
import FILE_PATHS


class FrameExtraction:
    def __init__(
        self,
        video_info,
        videos_path=FILE_PATHS.VIDEOS,
        path_save=FILE_PATHS.IMAGES,
        images_info_path=f"{FILE_PATHS.ECHONET}/images_info.csv",
    ):
        self.video_info = video_info
        self.path = videos_path
        self.path_save = path_save
        self.images_info_path = images_info_path

    def save_images(self):
        frame_info = pd.DataFrame(columns=["File", "X", "Y"])
        files = os.listdir(self.path)

        for file in files[:1000]:
            path_video = os.path.join(self.path, file)
            frames = self.video_info[self.video_info.FileName == file]["Frame"].unique()
            cap = cv2.VideoCapture(path_video)

            for frame in frames:
                landmarks = []
                name_img = f"{file[:-4]}_{frame}.png"
                path_img = os.path.join(self.path_save, name_img)
                coor = self.video_info[
                    (self.video_info.FileName == file)
                    & (self.video_info.Frame == frame)
                ]
                
                if (coor["X1"] > 112).any() or (coor["Y1"] > 112).any() or (coor["X2"] > 112).any() or (coor["Y2"] > 112).any():
                    print(f"Skipped file {file} because it contains coordinates greater than 112.")
                    continue
                
                puntos1 = coor.apply(lambda row: (int(row["X1"]), int(row["Y1"])), axis=1).tolist()
                puntos2 = coor.apply(lambda row: (int(row["X2"]), int(row["Y2"])), axis=1).tolist()
                landmarks = self.sort_points(puntos1, puntos2)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, img = cap.read()

                if ret:
                    for punto in landmarks:
                        new_row = {"File": name_img, "X": punto[0], "Y": punto[1]}
                        frame_info.loc[len(frame_info)] = new_row

                    cv2.imwrite(path_img, img)

        frame_info.to_csv(self.images_info_path, index=False)
        print("¡Extraction Done!")
        print(f"Path images: {self.path_save}")
        print(f"Path df: {self.images_info_path}")

    def sort_points(self, puntos1, puntos2):
        lands = np.concatenate((puntos1, puntos2), axis=0)

        centroid = np.mean(lands, axis=0)
        x = lands[:, 0]
        y = lands[:, 1]

        angles = np.arctan2(y - centroid[1], x - centroid[0])

        sorted_indices = np.argsort(angles)
        sorted_points = [lands[i] for i in sorted_indices]

        return sorted_points
