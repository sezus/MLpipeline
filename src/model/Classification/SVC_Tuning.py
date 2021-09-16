import os
import sys
sys.path.append('../..')
import yaml
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, make_scorer
from src.utils.utils_p import Upload_Download_Pickle,Model_Configs,Config_Paths
from src.model.Classification.Main.Base import Classifier
from os.path import join

class SVC_Tune(Classifier):
    def __init__(self):

        self.config = Config_Paths()
        self.processed_path = self.config.get_processed_path()
        self.model_path = self.config.get_model_path()
        self.metric_path = self.config.get_modelsOutputs_path()
        
        self.model_config = Model_Configs()
        self.input=self.model_config.get_inputFiles()
        self.model_params=self.model_config.get_models_params("SVM")
        self.parameters=self.set_model_params()
        self.number_of_cv_folds = self.model_config.get_number_of_cv_folds("SVM")
        self.grid_search_ = self.model_config.get_grid_search("SVM")
        self.average=self.model_config.get_scoring_type("SVM")

        self.model_name='SVM'
        self.model = SVC()

        self.log= 'SVC'  # __name__=projectA.moduleB
        self.pickle_file=Upload_Download_Pickle()

        #Metric_Plots    
        self.results='bests_SVM_classification.pckl'
        self.param1='param_gamma'
        self.param2='param_C'

    def set_model_params(self):
        param=self.model_params
        if param!=None:
            parameters=param['parameters']
            print('parameters', parameters)
            #parameters = {
            #    'C': param["C"],
            #    'gamma':param["gamma"],  
            #    'kernel': param["kernel"]             
            #}
        else:
            parameters = {
                'gamma':'auto'
            }

        return parameters

