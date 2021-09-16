import pickle
import numpy as np
import pandas as pd
import yaml
import os
import sys
from src.utils.utils_p import Upload_Download_Pickle, Config_Paths, Model_Configs
from src.model.Classification.Main.Base import Classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
sys.path.append('../..')
class Random_Forest_Tune(Classifier):
    def __init__(self):

        self.config = Config_Paths()
        self.processed_path = self.config.get_processed_path()
        self.model_path = self.config.get_model_path()
        self.metric_path = self.config.get_modelsOutputs_path()

        self.model_config = Model_Configs()
        self.input=self.model_config.get_inputFiles()
        self.model_params = self.model_config.get_models_params("RandomForest")
        self.parameters = self.set_model_params()
        self.grid_search_ = self.model_config.get_grid_search("RandomForest")
        self.number_of_cv_folds = self.model_config.get_number_of_cv_folds("RandomForest")
        self.average=self.model_config.get_scoring_type("RandomForest")
        
        self.model_name = 'RF'
        self.model = RandomForestClassifier(n_jobs=10, random_state=23)

        self.log = 'Random_Forest_Tune'  # __name__=projectA.moduleB
        self.pickle_file=Upload_Download_Pickle()


        #Metric_Plots    
        self.results='bests_RF_classification.pckl'
        self.param1='param_max_depth'
        self.param2='param_min_impurity_split'
   
    def set_model_params(self):
        param = self.model_params
        if (param != None):
            parameters=param['parameters']
            print('parameters', parameters)
            #parameters = {
            #    'max_depth': param["max_depth"],
            #    'min_impurity_split': [float(i) for i in param["min_impurity_split"]]
            #}
            print(parameters)
        else:
            parameters = {
                'max_depth': None,
                'min_impurity_split': None
            }
        return parameters




