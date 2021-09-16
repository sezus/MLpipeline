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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, make_scorer,roc_auc_score
#from src.utils.utils_p import save_dataset_pickle, download_pickle, Config_Paths, Model_Paths
from src.utils.utils_p import YamlParser,Config_Paths,Upload_Download_Pickle, Model_Configs, Feature_Selection_Configs, Logging_Config
from src.featureProcessing.Feature_Selection.Base import FeatureSelection

class Unigini(FeatureSelection):
    def __init__(self):
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.model_path=self.model_config.get_models_path()
        self.log_cfg=Logging_Config()  
      
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Unigini'  # __name__=projectA.moduleB
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
        self.cut_off_below=self.feature_selection_method_conf_details['cut_off_below']
        self.cut_off_change=self.feature_selection_method_conf_details['cut_off_change']
        self.model = RandomForestClassifier
        self.method = f_classif
        self.classification=self.feature_selection_conf.get_Classification()
        self.average=self.feature_selection_conf.get_Average()

        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Unigini'  # __name__=projectA.moduleB
        self.precondition=='Both_train_valid'
        self.pickle_file=Upload_Download_Pickle()      
        self.meta=self.pickle_file.download_pickle(self.processed_path,'meta')

    def filter_feature_selection(self,X_train, X_valid, y_train, y_valid):            
            print('meta',self.meta)
            print('X_train',X_train.shape)
            print('X_valid',X_valid.shape)
            print('y_train',y_train.shape)
            print('y_valid',y_valid.shape)
            meta,unigini=self.unigini_elimination(X_train, X_valid, y_train, y_valid,self.meta,self.cut_off_below,self.cut_off_change)
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'unigini',unigini)            
            selectedfeatures=self.meta[self.meta['STATUS']=='KEEP']['VARIABLE'].values        
            return selectedfeatures,meta

    def unigini_elimination(self,X_train,X_val,y_train,y_val,meta,cut_off_below,cut_off_change,variable_column='VARIABLE'):
        print('unigini elimination')
        unigini=pd.DataFrame()
        unigini.index.name= 'VARIABLE'
        meta_new=meta.copy()
        print('meta_new.shape',meta_new.shape)
        basic=meta_new[(meta_new[variable_column].isin(X_train.columns))].copy()
        basic=basic[(basic['STATUS']=='KEEP')]
        features_num,features_char=meta_features(basic)

        for var in features_num:
            data_x=X_train[[var]].values
            data_y=pd.DataFrame(y_train).values.ravel()
            mdl=XGBClassifier(max_depth=3,n_estimators=1)
            mdl.fit(data_x,data_y)
            data_y_predicted=mdl.predict_proba(data_x)
            unigini.loc[var,'TRAIN_ROC']= roc_auc_score(y_train,pd.DataFrame(data_y_predicted)[1])

            data_x=X_val[[var]].values
            data_y=pd.DataFrame(y_val).values.ravel()
            mdl=XGBClassifier(max_depth=3,n_estimators=1)
            mdl.fit(data_x,data_y)
            data_y_predicted=mdl.predict_proba(data_x)
            unigini.loc[var,'VAL_ROC']= roc_auc_score(y_val,pd.DataFrame(data_y_predicted)[1])

        unigini['TRAIN_GINI']=2*unigini['TRAIN_ROC']-1
        unigini['VAL_GINI']=2*unigini['VAL_ROC']-1
        unigini['CHANGE']=np.abs(unigini['TRAIN_GINI']-unigini['VAL_GINI'])/unigini['TRAIN_GINI']
        unigini['STATUS']=np.where(abs(unigini['TRAIN_GINI'])<cut_off_below,'DROP_BELOW','KEEP')
        unigini['STATUS']=np.where(abs(unigini['CHANGE'])>cut_off_change,'DROP_CHANGE',unigini['STATUS'])
        if unigini.index.name=='VARIABLE':
             unigini.reset_index(inplace=True)

        unigini_change_elimination=unigini[unigini['STATUS']=='DROP_CHANGE']['VARIABLE'].values
        unigini_below_elimination=unigini[unigini['STATUS']=='DROP_BELOW']['VARIABLE'].values
        meta['STATUS']=np.where(meta[variable_column].isin(unigini_change_elimination),'DROP_CHANGE',meta['STATUS'])
        meta['STATUS']=np.where(meta[variable_column].isin(unigini_below_elimination),'DROP_BELOW',meta['STATUS'])
      
        return meta,unigini

def meta_features(meta):
    features_num=meta[meta['TYPE']=='NUMERIC']['VARIABLE']
    features_char=meta[meta['TYPE']=='CHAR']['VARIABLE']
    return features_num,features_char 

   