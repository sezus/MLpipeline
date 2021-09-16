import os
import sys
sys.path.append('..') 
import yaml
import pandas as pd
import numpy as np
import pickle
from src.utils.utils_p import YamlParser,Upload_Download_Pickle,Config_Paths,Feature_Selection_Configs,Model_Configs,Create_Labels_Configs
from src.data.CollectData.CollectDataFromCSV import *
from src.preprocessing.Data_Preprocessing import *
from src.combineDatasets.Combine_Datasets import *
from src.featureProcessing.SplitDataset.Split_Train_Test import *
from src.featureProcessing.Feature_Selection.Wrapper import *
from src.featureProcessing.Feature_Selection.Hybrid import *
#from src.processing.Model import *
from src.model.Classification.Random_Forest_Tuning import *
from src.model.Classification.XGBoost_Tuning import *
from src.model.Classification.LightGBM_Tuning import *
from src.model.Classification.CatBoost_Tuning import *
from src.model.Classification.SVC_Tuning import *
from src.model.Classification.Voting import *


Collect_Data_From_CSV().upload_all_datasets() 
Data_Preprocessing().Preprocessing()  
Combine_Datasets().combine_dataset()
Split_Train_Test().split_train_testset()
Hybrid().featureSelection()

XGBoost_Tune().Model()
LightGBM_Tune().Model()
CatBoost_Tune().Model()
Random_Forest_Tune().Model()
Voting().Model()
SVC_Tune().Model()

# Create Model Metrics according to test sets
XGBoost_Tune().Model_Metrics('classification')
LightGBM_Tune().Model_Metrics('classification')
CatBoost_Tune().Model_Metrics('classification')
Random_Forest_Tune().Model_Metrics('classification')
SVC_Tune().Model_Metrics('classification')

# Plot Results of Grid Search
XGBoost_Tune().grid_plot()
LightGBM_Tune().grid_plot()
CatBoost_Tune().grid_plot()
Random_Forest_Tune().grid_plot()
SVC_Tune().grid_plot()
