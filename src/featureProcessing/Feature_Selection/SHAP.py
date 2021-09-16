import os
import sys
sys.path.append('..\..') 
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
import xgboost
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, make_scorer
#from src.utils.utils_p import save_dataset_pickle, download_pickle, Config_Paths, Model_Paths
from src.utils.utils_p import YamlParser,Config_Paths,Upload_Download_Pickle, Model_Configs, Feature_Selection_Configs, Logging_Config
from src.featureProcessing.Feature_Selection.Base import FeatureSelection

class SHAP(FeatureSelection):
    def __init__(self):
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.model_path=self.model_config.get_models_path()

        # Feature Selection Configs
        self.feature_selection_conf=Feature_Selection_Configs()
        self.input=self.feature_selection_conf.get_inputFiles()
        self.threshold = self.feature_selection_conf.get_Feature_Selection_Threshold()
        self.method_params=self.feature_selection_conf.get_Feature_Selection_Method()
        self.method_names = self.method_params['Name']
        self.model = RandomForestClassifier
        self.method= f_classif
        
        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'SHAP'  # __name__=projectA.moduleB
        self.precondition = 'Both_train_valid'
        self.pickle_file=Upload_Download_Pickle()   

    def filter_feature_selection(self,X_train, X_valid, y_train, y_valid,method,model,rand,threshold):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test,approximate=True)
            shap.summary_plot(shap_values, X_test, plot_type="bar")
            return selectedfeatures

   