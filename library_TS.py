# Common library
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))
import os
import numpy as np
print("NumPy version: {}". format(np.__version__))
import pandas as pd
print("pandas version: {}". format(pd.__version__))
from pandas.errors import EmptyDataError
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
print('matplotlib: {}'.format(matplotlib.__version__))
import glob
import re
from dateutil.parser import parse
import time

######################################################################################################################################################
######################################################################################################################################################

# TS fresh for TS classification
import tsfresh
from tsfresh import extract_features, select_features # For the classification
from tsfresh.utilities.dataframe_functions import impute # For the imputation
from tsfresh.feature_selection.relevance import calculate_relevance_table
print('TSfresh: {}'.format(tsfresh.__version__))

######################################################################################################################################################
######################################################################################################################################################

# For the ML algorithm
import sklearn
print("Sklearn version: {}".format(sklearn.__version__))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,recall_score,average_precision_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost
from xgboost import XGBClassifier
print("Xgboost version: {}".format(xgboost.__version__))

import lightgbm as lgb
print("Lightgbm version: {}".format(lgb.__version__))

