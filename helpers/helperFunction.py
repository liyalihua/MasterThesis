import pandas as pd
import numpy as np
import inspect #environment related helper function


# data related helper functions
def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data

def splitByRatio(df,trainRatio):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe: train and test
    """
    # indexing the dates
    df_desc = df.pivot(index='date',columns='tic',values='close')
    trainEndIdx = int(np.round(len(df_desc) * trainRatio,0)) # index training end
    trainStart = df_desc.index[0]
    trainEnd = df_desc.index[trainEndIdx]
    testStart = df_desc.index[trainEndIdx+1]
    testEnd = df_desc.index[-1]
    print(f'training periode: {trainStart} and {trainEnd}')
    print(f'training periode: {testStart} and {testEnd}')
    #split data
    train = data_split(df, trainStart,trainEnd) # split
    test = data_split(df, testStart,testEnd)
    return train,test

#environment related helper function


def get_attributes(env):
  """return a list of attributes given an environment"""
  attributes =  inspect.getmembers(env,lambda a:not(inspect.isroutine(a)))
  results = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
  return results

def get_method(env):
  """ return a list of methods given a environment"""
  method_list = [method for method in dir(env) if method.startswith('__') is False]
  return method_list
