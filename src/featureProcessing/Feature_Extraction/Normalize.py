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
class Normalize(FeatureExtraction):
    def __init__(self):
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.model_path=self.model_config.get_models_path()

        # Feature Selection Configs
        self.feature_extraction_conf=Feature_Extraction_Configs()
        self.feat_ext_conf=self.feature_extraction_conf.get_Feature_Extraction_Method_Config('Normalize')
        self.input=self.feature_extraction_conf.get_Method_inputFiles('Normalize')
        self.normalize_feature_list=self.feat_ext_conf['normalize_feature_list']
        self.extraction_list=self.feat_ext_conf['list']
        self.method='_norm'
        self.pickle_file=Upload_Download_Pickle()       
        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Normalize'  # __name__=projectA.moduleB

    def transform(self,X_train,X_train_list,meta,i):
        # normalize all dataset
        # X_train için:
        dataf=X_train.copy()
        self.normalize_feature_list=self.feat_ext_conf[i]['normalize_feature_list']
        print('meta',meta)
        #featurelist=meta_new[(meta_new['STATUS']=='KEEP')& (meta_new['TYPE']=='NUMBER')]
        print(dataf.dtypes)
        if self.normalize_feature_list != None:            
            # Normalize all continious columns burda sadece verilen list ile verilen kolonları normalize eder
            columns_new=dataf.columns[dataf.columns.isin(self.normalize_feature_list)]
            print('columns_new',columns_new)
            dataf_add = self.normalize_all(dataf, columns_new)
            print('dataf_add: ',dataf_add.shape)
            print('dataf_add: ',dataf_add.head())
            dataf_new=pd.merge(dataf,dataf_add, how='left',left_index=True, right_index=True)
            print('dataf_new: ',dataf.shape)
            print('dataf_new: ',dataf.head())
        else:
            raise ValueError('normalize_feature_list config is missing!') 
        # X_train_list için: 
        data_list=[]        
        for datafrm in X_train_list:
            print('dataf_new.shape',datafrm.shape)
            if self.normalize_feature_list != None:
                # Normalize all continious columns burda sadece verilen list ile verilen kolonları normalize eder
                columns_new=datafrm.columns[datafrm.columns.isin(self.normalize_feature_list)]
                print('columns_new',columns_new)             
                datafrm_add = self.normalize_all(datafrm, columns_new)
                print('datafrm_add: ',datafrm_add.shape)
                #datafrm_new=pd.concat([datafrm,datafrm_add])
                datafrm_new=pd.merge(datafrm,datafrm_add, how='left',left_index=True, right_index=True)
                print('datafrm_new: ',datafrm_new.shape)
                data_list.append(datafrm_new)
            else:
                raise ValueError('normalize_feature_list config is missing!')       
            
        return dataf_new, data_list  

    def normalize_all(self, dataf, feature_list):
        datafn = dataf.copy()
        for i in feature_list:
            datafn[i] = self.normalize(datafn[i].values)
        return datafn.add_suffix(self.method)
    
    def normalize(self, val):
        val = (val-np.min(val))/(np.max(val)-np.min(val))
        return val

