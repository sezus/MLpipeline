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
from src.utils.utils_p import YamlParser,Config_Paths,Upload_Download_Pickle, Model_Configs, Feature_Extraction_Configs, Logging_Config
from src.featureProcessing.Feature_Extraction.Base import FeatureExtraction
class Rolling(FeatureExtraction):
    def __init__(self):
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.model_path=self.model_config.get_models_path()

        # Feature Selection Configs
        self.feature_extraction_conf=Feature_Extraction_Configs()
        self.feat_ext_conf=self.feature_extraction_conf.get_Feature_Extraction_Method_Config('Roll')
        self.extraction_list=self.feat_ext_conf['list']
        self.input=self.feature_extraction_conf.get_Method_inputFiles('Roll')
        self.extraction=self.feature_extraction_conf
        self.method='mean'
        self.diff='2'
        self.pickle_file=Upload_Download_Pickle()       
        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Rolling'  # __name__=projectA.moduleB

    def transform(self,X_train,X_train_list,meta,i):
        # normalize all dataset
        # X_train için:
        dataf=X_train.copy()
        print('dataf.dtypes',dataf.dtypes)
        self.method=self.feat_ext_conf[i]['method']
        self.diff=self.feat_ext_conf[i]['diff']
        featurelist=meta[(meta['STATUS']=='KEEP')& (meta['TYPE']=='NUMBER')]
        dataf_add = self.roll_all(dataf,featurelist['VARIABLE'],self.method,self.diff)
        print('dataf_add: ',dataf_add.shape)
        print('dataf_add: ',dataf_add.head())
        dataf_new=pd.merge(dataf,dataf_add, how='left',left_index=True, right_index=True)
        print('dataf_new: ',dataf.shape)
        print('dataf_new: ',dataf.head())

        # X_train_list için: 
        data_list=[]        
        for datafrm in X_train_list:
            print('dataf_new.shape',dataf_new.shape)      
            # Normalize all continious columns burda sadece verilen list ile verilen kolonları normalize eder
            datafrm_add = self.roll_all(datafrm, featurelist['VARIABLE'],self.method,self.diff)
            print('datafrm_add: ',datafrm_add.shape)
            #datafrm_new=pd.concat([datafrm,datafrm_add])
            datafrm_new=pd.merge(datafrm,datafrm_add, how='left',left_index=True, right_index=True)
            print('datafrm_new: ',datafrm_new.shape)
            data_list.append(datafrm_new)

        return dataf_new, data_list  
        
    @staticmethod
    def roll_all( dataf, feature_list,method,diff):
        datafn = dataf.copy()
        for i in feature_list:
            if method=='_roll_std':
                datafn[i] = datafn[i].rolling(diff).std()
            elif method=='_roll_mean':
                datafn[i] = datafn[i].rolling(diff).mean()
            elif method=='_roll_sum':
                datafn[i] = datafn[i].rolling(diff).sum()
            elif method=='_roll_sum':
                datafn[i] = datafn[i].rolling(diff).diff()  
        return datafn.add_suffix(method)
    

