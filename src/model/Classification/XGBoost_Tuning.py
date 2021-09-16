import os
import sys

sys.path.append('../..')
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from src.utils.utils_p import Upload_Download_Pickle, Model_Configs, Config_Paths
from src.model.Classification.Main.Base import Classifier
from os.path import join


class XGBoost_Tune(Classifier):
    def __init__(self):

        self.config = Config_Paths()
        self.processed_path = self.config.get_processed_path()
        self.model_path = self.config.get_model_path()
        self.metric_path = self.config.get_modelsOutputs_path()
        
        self.model_config = Model_Configs()
        self.input=self.model_config.get_inputFiles()
        self.model_params = self.model_config.get_models_params("XGboost")
        self.parameters = self.set_model_params()
        self.number_of_cv_folds = self.model_config.get_number_of_cv_folds("XGboost")
        self.grid_search_ = self.model_config.get_grid_search("XGboost")
        self.average=self.model_config.get_scoring_type("XGboost")

        self.model_name = 'XGboost'
        self.model = XGBClassifier(n_jobs=10, random_state=23)   

        self.log= 'XGboost'  # __name__=projectA.moduleB
        self.pickle_file=Upload_Download_Pickle()    

        #Metric_Plots     
        self.results='bests_XGboost_classification.pckl'
        self.param1='param_max_depth'
        self.param2='param_min_child_weight'   


    def set_model_params(self):
        param = self.model_params
        if (param != None):
            parameters=param['parameters']
            #parameters = {
            #    'max_depth': param["max_depth"],
            #    'min_child_weight': [float(i) for i in param["min_child_weight"]]
            #}
            print(parameters)
        else:
            parameters = {
                'max_depth': None,
                'min_impurity_split': None
            }

        return parameters
