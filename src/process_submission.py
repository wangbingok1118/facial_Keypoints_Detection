# -*_ coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import config



def main():
    originalDataPath = '/home/hduser0539/projectsData/privateProjectData/facial-keypoints-detection'
    testResultFile = 'test_result.csv'
    IdlookFile = 'IdLookupTable.csv'
    testResutlDf = pd.read_csv(os.path.join(originalDataPath, testResultFile))
    IdLookDf = pd.read_csv(os.path.join(originalDataPath, IdlookFile))
    subMissionDf = pd.read_csv(os.path.join(originalDataPath, 'SampleSubmission.csv'))
    for row in IdLookDf.iterrows():
        row_index = row[0]
        row_item = row[1]
        value = testResutlDf.get_value(index=int(row_item['ImageId']) - 1, col=row_item['FeatureName'])
        value = "%.4f" % (value)
        subMissionDf.loc[subMissionDf['RowId'] == row_item['RowId'], 'Location'] = value
    print(subMissionDf.head())

    subMissionDf.to_csv(os.path.join(originalDataPath, 'SampleSubmission_result.csv'), index=False)
    pass


if __name__ == '__main__':
    main()