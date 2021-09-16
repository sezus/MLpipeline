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
#from src.utils.utils_p import save_dataset_pickle, download_pickle, Config_Paths, Model_Paths
from src.utils.utils_p import YamlParser,Config_Paths,Upload_Download_Pickle, Model_Configs, Feature_Selection_Configs, Logging_Config
from src.featureProcessing.Feature_Selection.Base import FeatureSelection

class Hybrid(FeatureSelection):
    def __init__(self):
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.model_path=self.model_config.get_models_path()       
        self.log_cfg=Logging_Config()   
        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Hybrid'  # __name__=projectA.moduleB
        self.precondition = 'Both_train_valid'

        # Feature Selection Configs
        self.feature_selection_conf=Feature_Selection_Configs()

        # Method config 
        self.feature_selection_method_conf =self.feature_selection_conf.get_FeatureSelection_method_conf()
        self.feature_selection_method_conf_details = self.feature_selection_method_conf[self.log]        

         # General config
        self.input=self.feature_selection_conf.get_inputFiles()        
        self.createmetadata=self.feature_selection_method_conf_details['Create_metadata']
       
        # Feature Selection Configs
        self.input=self.feature_selection_conf.get_inputFiles()
        self.threshold = self.feature_selection_method_conf_details['Feature_Selection_Threshold']
        self.model = RandomForestClassifier
        self.method = f_classif
        self.rand = self.feature_selection_method_conf_details['random']
        self.classification=self.feature_selection_method_conf_details['Classification']
        self.average=self.feature_selection_method_conf_details['Average']


        self.pickle_file=Upload_Download_Pickle()   
        self.metaconf=self.feature_selection_conf.get_metaFile()
        print('self.metaconf: ',str(self.metaconf[0]))
        self.meta=self.pickle_file.download_pickle(self.processed_path,'meta')

    def filter_feature_selection(self,X_train, X_valid, y_train, y_valid):
            f1_train_all=[]
            f1_valid_all=[]
            column_names=[]  
            print('self.meta',self.meta)   
            meta_new=self.meta[self.meta['VARIABLE'].isin(X_train.columns)].copy()
            meta_new=meta_new[meta_new['STATUS']=='KEEP'].copy()
            columns=X_train.columns[X_train.isna().sum()==0]
            columns_new=meta_new[(meta_new['TYPE']!='CHAR')&(meta_new['VARIABLE'].isin(columns))]
            
            X_train_new=X_train[columns_new['VARIABLE']]       
            for k in range(1,X_train_new.shape[1]):
                Column_Names = self.filtering(X_train_new,y_train,self.method,k)
                #print('Column_Names',Column_Names)
                #print('Column_Names',X_train[X_train.columns[Column_Names]].head())
                f1_train, f1_valid = self.train_(X_train_new[X_train_new.columns[Column_Names]], y_train, X_valid[X_train_new.columns[Column_Names]], y_valid, self.model,self.rand,self.average)
                f1_train_all.append(f1_train)
                f1_valid_all.append(f1_valid)
                column_names.append(X_train_new.columns[Column_Names])           
                self.logger.warning('filter selection wrapper column_names %s with f_train %s and f_valid %s' %( X_train_new.columns[Column_Names], f1_train, f1_valid) )
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'f1_train_all',f1_train_all)
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,' f1_valid_all',f1_valid_all)
            selectedfeatures,meta=self.select_feature_performance_increase_bigger_than_threshold(self.meta,f1_train_all,f1_valid_all,column_names,self.threshold) 
            return selectedfeatures,meta

   