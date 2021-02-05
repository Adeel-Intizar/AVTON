import cv2
import os
# import matplotlib.pyplot as plt
# import copy
# import numpy as np
import json
import tensorflow as tf

# from avton.pose.src import model
from .src import util
from .src.body import Body



class Keypoints:
    
    def __init__(self):
        
        self.model = None
        self.weights_path = ""
        self.modelLoaded = False
    
    
    def load_model(self, model_path:str = None):

        if model_path is None:
            
            path = 'avton/weights/keypoints_model.pth'
            if os.path.isfile(path):
                print('Found Existing Weights File...\nLoading Existing File...')
                self.weights_path = path
            else:
                print('Downloading Weights File...\nPlease Wait...')
                self.weights_path = tf.keras.utils.get_file('keypoints_model.pth',
                'https://github.com/Adeel-Intizar/AVTON/releases/download/1/keypoints_model.pth',
                cache_subdir = 'weights/', cache_dir = 'avton')
        else:
            if os.path.isfile(model_path):
                self.weights_path = model_path
            else:
                raise FileNotFoundError ("Weights File Doesn't Exist at Provided Path. Please Provide Valid Path.")

        self.model = Body(self.weights_path)
        self.modelLoaded = True
        
        
    def Detect_From_Image(self, input_image:str, output_image:str = None, json_keypoints:str = None):
        
        """
        input_image: path to the input image jpg or jpeg
        output_image: path to save the output image e.g. C:/Desktop/output.jpg
        json_keypoints: path to save the output json file, if not specified default='avton/outputs/keypoints.json'
        """
        
        if not self.modelLoaded:
            raise RuntimeError("Model is not Loaded, Please call load_model() First")
        
        oriImg = cv2.imread(input_image)
        candidate, subset = self.model(oriImg)
        
        if output_image is not None:
            canvas = util.draw_bodypose(oriImg, candidate, subset)
            cv2.imwrite(output_image, canvas)
            
        if json_keypoints is None:
            json_keypoints = "keypoints.json"
        
        keypoints = []
        for i in range(18):
            for n in range(len(subset)):
                index = int(subset[n][i])
                if index == -1:
                    continue
                keypoints.extend(candidate[index][0:3])
        
        
        oneperson = { "face_keypoints": [],
                      "pose_keypoints": keypoints,
                      "hand_right_keypoints": [],
                      "hand_left_keypoints":[]}
        
        # print(f"Total keypoints: {len(keypoints)}")
        # print(keypoints)
        
        joints_json =  { "version": 1.0, "people": [oneperson] }
        with  open(json_keypoints, 'w') as joint_file:
                json.dump(joints_json, joint_file)
 
