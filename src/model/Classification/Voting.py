import os
import sys
sys.path.append('../..')
import yaml
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, make_scorer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.utils.utils_p import Upload_Download_Pickle,Model_Configs,Config_Paths
from src.model.Classification.Main.Base import Classifier
from os.path import join

class Voting(Classifier):

    def __init__(self):

        self.config = Config_Paths()
        self.processed_path = self.config.get_processed_path()
        self.model_path = self.config.get_model_path()
        self.metric_path = self.config.get_modelsOutputs_path()
        
        self.model_config = Model_Configs()
        self.input=self.model_config.get_inputFiles()
        self.model_params=self.model_config.get_models_params("Voting")
        self.parameters= self.set_model_params()
        self.grid_search_=False

        self.class_model = Classifier()
        self.model_name = 'Voting'
        self.model = VotingClassifier(estimators = self.parameters[-1], voting='hard')

        self.log= 'Voting'  # __name__=projectA.moduleB
        self.pickle_file=Upload_Download_Pickle()

    def set_model_params(self):
        param = self.model_params
        if (param != None):
            models_list = []
            for i in param["model_list"]:
                model_name = i +'_finalized_model.pckl'
                print(model_name)
                print(self.model_path)
                model=Upload_Download_Pickle().download_pickle(self.model_path,model_name)
                models_list.append((i,model))
        self.model_list=models_list

        return param,models_list
