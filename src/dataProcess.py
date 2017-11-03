# -*- coding: utf-8 -*-
import config
import numpy as np
import os
import sys
import pandas as pd


def loadData(fileName=None,test=False):
    """
    the function used to read csv file
    :param fileName:
    :param test:
    :return:
    """
    df = pd.read_csv(fileName)  # load pandas dataframe
    df['Image'] = df['Image'].apply(lambda img : np.fromstring(img,sep=' '))
    print('%s describe info is :'%(fileName))
    print(df.count())
    df.dropna()

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    pass

def main():
    pass
if __name__ == '__main__':
    main()