# -*- coding: utf-8 -*-


import os
import sys
import pandas as pd
import numpy as np


"""
configure
"""

# data path
originalDataPath = '/home/hduser0539/projectsData/privateProjectData/facial-keypoints-detection'
trainFilePath = os.path.join(originalDataPath,'training.csv') # use temp file
testFilePath = os.path.join(originalDataPath,'test.csv')  # use tmp file

# validataion data size
validataionSize = 0.2
mini_batch_size=32

image_channels=1
label_numbers=30
filter_size = 3
dropout_rate = 0.5
train_epcohs = 100
# # log path
# logFilePath = '../data'
# logFileName = 'facial_keypoints_detection.log'
# logPathAndName = os.path.join(logFilePath,logFileName)


IMAGE_SIZE = 96  # face images 96x96 pixels