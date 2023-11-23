import cv2
import os
import numpy as np
import pandas as pd
import FILE_PATHS

class FrameExtraction:
    def __init__ (
        self, 
        video_info, 
        videos_path=FILE_PATHS.VIDEOS, 
        path_save=FILE_PATHS.IMAGES,
        images_info_path=f'{FILE_PATHS.ECHONET}/images_info.csv'
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
                landmarks = self.sort_points(puntos1, puntos2)
                
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, img = cap.read()

                if ret:
                    if num_landmarks == None:
                        for punto in landmarks:
                            new_row = {'File': name_img, 'X': punto[0], 'Y': punto[1]}
                            frame_info.loc[len(frame_info)] = new_row
                    else:
                        frame_info = self.optimalLandmarks(landmarks, frame_info, name_img, num_landmarks)
                    cv2.imwrite(path_img, img)
        
        frame_info.to_csv(self.images_info_path, index=False)
        print('¡Straction Done!')
        print('Path images: ', self.path_save)
        print('Path df: ', self.images_info_path)
        
        
        
    def optimalLandmarks(self, sorted_points, dataframe, name, num_landmarks):
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
        
            angles.append(angle)
            landmark.append(b)
            
        sorted_index = np.argsort(angles)
        optimal_lands = [sorted_points[i] for i in sorted_index]
        
        
        for punto in optimal_lands[:num_landmarks]:
            new_row = {'File': name, 'X': punto[0], 'Y': punto[1]}
            dataframe.loc[len(dataframe)] = new_row
            
        return dataframe
    
    
    def sort_points(self, puntos1, puntos2):
        lands = np.concatenate((puntos1, puntos2), axis=0)
        
        centroid = np.mean(lands, axis=0)
        x = [coord[0] for coord in lands]
        y = [coord[1] for coord in lands]
                
        angles = np.arctan2(y - centroid[1], x - centroid[0])

        sorted_indices = np.argsort(angles)
        sorted_points = [lands[i] for i in sorted_indices]
        
        return sorted_points
        