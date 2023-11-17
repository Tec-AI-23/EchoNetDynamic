import cv2
import os
import numpy as np
import pandas as pd


class FrameExtraction:
    def __init__ (
        self, 
        video_info, 
        videos_path='../EchoNet-Dynamic/videos/', 
        path_save='../EchoNet-Dynamic/data/images/',
        images_info_path='../EchoNet-Dynamic/images_info.csv'
        ):

        self.video_info = video_info
        self.path = videos_path
        self.path_save = path_save
        self.images_info_path = images_info_path

    def save_images(self, num_landmarks=None):
        frame_info = pd.DataFrame(columns=['File', 'X', 'Y'])
        files = os.listdir(self.path)

        for file in files[:2]:
            path_video = os.path.join(self.path, file)
            frames = self.video_info[self.video_info.FileName == file]["Frame"].unique()
            cap = cv2.VideoCapture(path_video)

            for frame in frames:
                landmarks = []
                name_img = file[:-4] + '_' + str(frame) + '.jpeg'
                path_img = os.path.join(self.path_save, name_img)
                coor = self.video_info[(self.video_info.FileName == file) & (self.video_info.Frame == frame)]
                puntos1 = [(int(row['X1']), int(row['Y1'])) for index, row in coor.iterrows()]
                puntos2 = [(int(row['X2']), int(row['Y2'])) for index, row in coor.iterrows()]
                landmarks = np.concatenate((puntos1, puntos2), axis=0)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, img = cap.read()

                if ret:
                    if num_landmarks == None:
                        for punto in landmarks:
                            new_row = {'File': name_img, 'X': punto[0], 'Y': punto[1]}
                            frame_info.loc[len(frame_info)] = new_row
                    else:
                        frame_info = self.optimalLandmarks(landmarks, frame_info, name_img)
                    cv2.imwrite(path_img, img)
        
        frame_info.to_csv(self.images_info_path, index=False)
        print('¡Straction Done!')
        print('Path images: ', self.path_save)
        print('Path df: ', self.images_info_path)
        
        
        
    def optimalLandmarks(self, landmarks, dataframe, name):
        puntos = []
        puntos.append(landmarks[0])
        puntos.append(landmarks[-1])
        puntos.append(landmarks[len(landmarks) // 4])
        puntos.append(landmarks[len(landmarks) // 2])
        puntos.append(landmarks[(len(landmarks)*3) // 4])
        
        for punto in puntos:
            new_row = {'File': name, 'X': punto[0], 'Y': punto[1]}
            dataframe.loc[len(dataframe)] = new_row
            
        return dataframe