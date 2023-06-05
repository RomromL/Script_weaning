# Common library
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))
import numpy as np
print("NumPy version: {}". format(np.__version__))
import pandas as pd
print("pandas version: {}". format(pd.__version__))
import time
import matplotlib
import matplotlib.pyplot as plt
print('matplotlib: {}'.format(matplotlib.__version__))
import seaborn as sns
print('seaborn: {}'.format(matplotlib.__version__))
import math
from collections import Counter
from sklearn import preprocessing

######################################################################################################################################################
######################################################################################################################################################

# ML algorithm library
import sklearn
print("Sklearn version: {}".format(sklearn.__version__))
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler # For the standardization
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, plot_roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve, classification_report, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier

from imblearn.metrics import sensitivity_specificity_support

import lightgbm as lgb
print("LightGBM: {}".format(lgb.__version__))

import xgboost as xgb
from xgboost import XGBClassifier
print("XGBoost: {}".format(xgb.__version__))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
print("TensorFlow: {}".format(tf.__version__))

from pyod.models.iforest import IForest #outliers detection

import pandasgui # Interface to make some statistical analysis

from scipy.stats import shapiro  #To make shapiro Wilk test

######################################################################################################################################################
######################################################################################################################################################

# Feature exploration
import shap
print("SHAP: {}".format(shap.__version__))

######################################################################################################################################################
######################################################################################################################################################

# Imputation
from sklearn.impute import KNNImputer # KNN method for missing value
from sklearn.experimental import enable_iterative_imputer # For MICE method
from sklearn.impute import IterativeImputer # For MICE method
from sklearn import linear_model # For MICE method

######################################################################################################################################################
######################################################################################################################################################

# Missing value library
import missingno as msno # Missing values management
print('missingno: {}'.format(msno.__version__))
import matplotlib.patches as mpatches

######################################################################################################################################################
######################################################################################################################################################

# Oversampling
import imblearn
from imblearn.over_sampling import SMOTE, SMOTENC #Oversampling
from imblearn.under_sampling import RandomUnderSampler #Undersampling
from imblearn.pipeline import Pipeline
print("imblearn: {}".format(imblearn.__version__))

from tensorflow_addons import losses

######################################################################################################################################################
######################################################################################################################################################

import joblib # Save the model to reuse it later
print("joblib: {}".format(joblib.__version__))

######################################################################################################################################################
######################################################################################################################################################

import tableone
from tableone import TableOne, load_dataset #Stats descriptive
print("tableone: {}".format(tableone.__version__))

from warnings import filterwarnings # Delete the warning message
filterwarnings('ignore')

