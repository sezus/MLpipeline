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

class FeatureExtraction():
    def __init__(self):
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.model_path=self.model_config.get_models_path()

        # Feature Selection Configs
        self.feature_extraction_conf=Feature_Extraction_Configs()
        self.input=self.feature_extraction_conf.get_inputFiles()
        self.feat_ext_conf=self.feature_extraction_conf.get_Feature_Extraction_Method_Config('Normalize')
        self.extraction_list=self.feat_ext_conf['list']
        self.pickle_file=Upload_Download_Pickle()       
        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Normalize'  # __name__=projectA.moduleB
        self.metaconf=self.feature_extraction_conf.get_metaFile()
        print('self.metaconf: ',str(self.metaconf[0]))
        self.meta=self.pickle_file.download_pickle(self.processed_path,'meta')
        print('self.meta',self.meta)

    def featureExtraction(self):
        
        self.logger = logging.getLogger(self.log)  # __name__=projectA.moduleB
        # Download Train Dataframes for Feature Selection:
        #X_trains=Upload_Download_Pickle().download_pickle(self.processed_path,'X_trains')
        #y_trains=Upload_Download_Pickle().download_pickle(self.processed_path,'y_trains')
        # Download Train List for Feature Selection:
        #X_trains_list=download_pickle(processed_path,'X_trains_list')
        #y_trains_list=download_pickle(processed_path,'y_trains_list')
        print('self.extraction_list',self.extraction_list)
        for i in self.extraction_list:
            self.input=self.feat_ext_conf[i]['Input']
            print('self.input',self.input)
            X_trains,X_test,X_trains_list,X_test_list=self.pickle_file.download_pickle_files(self.processed_path,self.input)
            meta=self.create_meta_data(X_trains)
            #meta=self.pickle_file.download_pickle_files(self.processed_path,meta)
            #yeni oluşturulmuş featureları dataset'e ekle
            #Create meta data         
            #if isinstance(self.meta, pd.DataFrame):
            #    print('metadata already created.')            
            #else:            
            #    print('metadata :',self.meta)
            #    print('Create new metadata and save:')
            #    self.meta=self.create_meta_data(X_trains)
            #    Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'meta',self.meta)
        
            X_trains_ext,X_trains_list_ext=self.transform(X_trains,X_trains_list,meta,i)
            X_test_ext,X_test_list_ext=self.transform(X_test,X_test_list,meta,i)
            # mevcut halini save et        
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'X_trains_ext'+str(i),X_trains_ext)
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'X_trains_list_ext'+str(i),X_trains_list_ext)
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'X_test_ext'+str(i),X_test_ext)
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'X_test_list_ext'+str(i),X_test_list_ext)

    def transform_(self):
        self.logger = logging.getLogger(self.log)  # __name__=projectA.moduleB
        # Download Train Dataframes for Feature Selection:
        #X_trains=Upload_Download_Pickle().download_pickle(self.processed_path,'X_trains')
        #y_trains=Upload_Download_Pickle().download_pickle(self.processed_path,'y_trains')
        # Download Train List for Feature Selection:
        #X_trains_list=download_pickle(processed_path,'X_trains_list')
        #y_trains_list=download_pickle(processed_path,'y_trains_list')
        X_trains,X_test,X_trains_list,X_tests_list,meta=self.pickle_file.download_pickle_files(self.processed_path,self.input)
        #yeni oluşturulmuş featureları dataset'e ekle

        X_trains,X_trains_list=self.transform(X_trains,X_trains_list)
        # mevcut halini save et
        Upload_Download_Pickle().save_dataset_pickle(self.processed_path,X_trains,X_trains_list)

    @staticmethod    
    def create_meta_data(data):
        feature_type=np.where(data.dtypes=='float64', 'NUMERIC','CHAR')
        feature_type=np.where(data.dtypes=='int64', 'NUMERIC',feature_type)
        metadata={'VARIABLE': data.columns, 'TYPE':feature_type, 'STATUS':'KEEP'}
        meta=pd.DataFrame(metadata)
        return meta

