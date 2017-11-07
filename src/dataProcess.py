# -*- coding: utf-8 -*-
import config
import numpy as np
import os
import sys
import pandas as pd
import sklearn.utils


def loadData(fileName=None,test=False):
    """
    the function used to read csv file
    :param fileName:
    :param test:
    :return:
    """
    dataframe = pd.read_csv(fileName)
    feature_cols = dataframe.columns[:-1]  # all but image column
    # transform image space-separated pixel values to normalized pixel vector
    dataframe['Image'] = dataframe['Image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)
    dataframe = dataframe.dropna()  # drop entries w/NaN entries

    # get all image vectors and reshape to a #num_images x image_size x image_size x channels tensor
    X = np.vstack(dataframe['Image'])
    X = X.reshape(-1, config.image_size, config.image_size, 1)

    if not test:
        # get label features and scale pixel coordinates by image range
        # because image 96*96, so feature value / 96
        y = dataframe[feature_cols].values / float(config.image_size)
        # permute (image, label) pairs for training
        X, y = sklearn.utils.shuffle(X, y)
    else:
        y = None
    return X, y
    pass
#
# def printFun(str=None,model=False):
#     if not model:
#         print(str)
#     if model:
#         if(os.path.exists(config.logPathAndName) == True):
#             os.

def main():
    pass
if __name__ == '__main__':
    main()