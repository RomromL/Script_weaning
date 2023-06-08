# Script_weaning
This repository contains all the script used for the EXTUB-IA project, an algorithm to predict success weaning in critical care.

Description of the different script : 
- 3d_graph_script : For 3D graph modelisation only
- Feature_extraction : For extraction of the features from the time series with differents options, based on FRESH method.
- Interval_test_script : To train the machine learning for differents time series intervals (48H, 24H, 12H, 2H) and determine the best interval.
- Only_discrete_script : To train the machine learning only for the discrete variables
- Only_TS_script : To train the machine learning only for the time series after feature extraction
- Statistic_script : To obtain the table of population characteristics (called Table one).
- TS_management_script : Cleaning the differents time series files (remove useless lines ...)

All the packages needed to run the differents jupyter notebook are contained in : library_discrete.py and library_TS.py

The database are not included in this repository (private file). 

If you want to use one of this script, please contact the author : lombardi.romain@gmail.com

