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
trainFilePath = os.path.join(originalDataPath,'training.csv')
testFilePath = os.path.join(originalDataPath,'test.csv')
onlyGetColumnsFile = os.path.join(originalDataPath,'tmpTraining.csv')
# test result save path
testResultFileName = os.path.join(originalDataPath,'test_result.csv')

# submission
IdLookupTable = 'IdLookupTable.csv'
IdLookupTableFile = os.path.join(originalDataPath,IdLookupTable)
SampleSubmission = 'SampleSubmission.csv'
SampleSubmissionFile = os.path.join(originalDataPath,SampleSubmission)

# validataion data size
validataionSize = 0.2
mini_batch_size=32

image_channels=1
image_size = 96  # face images 96x96 pixels
label_numbers=30
filter_size = 3
dropout_rate = 0.5
train_epcohs = 1000

#model save path
modelPath = '../data/model'
modelName = 'cnnModel'
# # log path
# logFilePath = '../data'
# logFileName = 'facial_keypoints_detection.log'
# logPathAndName = os.path.join(logFilePath,logFileName)


