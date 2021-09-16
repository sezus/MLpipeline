import os
import sys
sys.path.append('../..')
import yaml
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from src.utils.utils_p import Upload_Download_Pickle, Model_Configs, Config_Paths
from src.model.Classification.Main.Base import Classifier
from os.path import join


class CatBoost_Tune(Classifier):
    def __init__(self):

        self.config = Config_Paths()
        self.processed_path = self.config.get_processed_path()
        self.model_path = self.config.get_model_path()
        self.metric_path = self.config.get_modelsOutputs_path()

        self.model_config = Model_Configs()
        self.input=self.model_config.get_inputFiles()
        self.model_params = self.model_config.get_models_params("Catboost")
        self.number_of_cv_folds = self.model_config.get_number_of_cv_folds("Catboost")
        self.grid_search_ = self.model_config.get_grid_search("Catboost")
        self.parameters = self.set_model_params()
        self.average=self.model_config.get_scoring_type("Catboost")

        #self.class_model = Classifier()
        self.model_name = 'Catboost'
        self.model = CatBoostClassifier(learning_rate=0.1, n_estimators=100)

        self.log = 'CatBoost_Tune'  # __name__=projectA.moduleB
        self.pickle_file=Upload_Download_Pickle()   
        
        #Metric_Plots  
        self.results='bests_Catboost_classification.pckl'
        self.param1='param_depth'
        self.param2='param_l2_leaf_reg'

    def set_model_params(self):
        param = self.model_params
        if (param != None):
            parameters=param['parameters']
            print('parameters', parameters)
            #parameters = {
            #    'depth': param["depth"],
            #    'l2_leaf_reg': [float(i) for i in param["l2_leaf_reg"]]
            #}
            print(parameters)
        else:
            parameters = {
                'max_depth': None,
                'min_impurity_split': None
            }

        return parameters
