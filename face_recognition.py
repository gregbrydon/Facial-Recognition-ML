#face_recognition.py

import warnings
warnings.filterwarnings('ignore')

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random, os
import pandas as pd

filepath = './fairface/train'
file_name = os.listdir(f"{filepath}")
for i in range(len(file_name)):
    file_name[i] = f"train/{file_name[i]}"

labels_df = pd.read_csv('fairface_label_train.csv')

df = labels_df[labels_df['file'].isin(file_name)]
df.to_csv('labels.csv')
print(df.head())
