import os
import sys
sys.path.append('..\..') 
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, make_scorer
from src.utils.utils_p import YamlParser,Config_Paths,Upload_Download_Pickle, Model_Configs, Feature_Selection_Configs, Logging_Config

class FeatureExtraction():
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
        self.methods = [chi2, f_classif]
        self.method_names = ['chi2','f_classif']
        self.model = RandomForestClassifier
        self.method=f_classif
        self.classification=self.feature_selection_conf.get_Classification()
        self.average=self.feature_selection_conf.get_Average()

        self.pickle_file=Upload_Download_Pickle()       
        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Feature_Selection'  # __name__=projectA.moduleB
    
    