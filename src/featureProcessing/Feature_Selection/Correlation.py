import os
import sys
sys.path.append('..\..') 
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, make_scorer
#from src.utils.utils_p import save_dataset_pickle, download_pickle, Config_Paths, Model_Paths
from src.utils.utils_p import YamlParser,Config_Paths,Upload_Download_Pickle, Model_Configs, Feature_Selection_Configs, Logging_Config
from src.featureProcessing.Feature_Selection.Base import FeatureSelection

class Correlation(FeatureSelection):
    def __init__(self):
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.model_path=self.model_config.get_models_path()
        self.log_cfg=Logging_Config()  
      
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Correlation'  # __name__=projectA.moduleB
        self.precondition='All_train_data'

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
        self.cut_off_corr=self.feature_selection_method_conf_details['cut_off_corr']
        self.size_coeff=self.feature_selection_method_conf_details['size_coeff']
        self.model = RandomForestClassifier
        self.method = f_classif
        self.classification=self.feature_selection_conf.get_Classification()
        self.average=self.feature_selection_conf.get_Average()

        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Correlation'  # __name__=projectA.moduleB
        self.precondition=='All_train_data'
        self.pickle_file=Upload_Download_Pickle()      
        self.meta=self.pickle_file.download_pickle(self.processed_path,'meta')
        self.unigini=self.pickle_file.download_pickle(self.processed_path,'unigini')

    def filter_feature_selection(self,X_trains,y_trains):
            meta,df_corr=self.corr_elimination (X_trains, self.meta,self.unigini)
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'df_corr',df_corr) 
            print('meta son: ',meta)
            selectedfeatures=meta[meta['STATUS']=='KEEP']['VARIABLE']        
            return selectedfeatures,meta

    def corr_elimination(self,X_train,meta,unigini,cut_off_corr=0.95,size_coeff=0.1):    
            #print('meta',meta)  
            meta_new=meta.copy()
            if meta_new.index.name=='VARIABLE':
                    feature_num=meta_new[(meta_new['TYPE']=='NUMERIC') & (meta_new['STATUS']=='KEEP') ].index.tolist()
            else:
                    feature_num=meta_new[(meta_new['TYPE']=='NUMERIC') & (meta['STATUS']=='KEEP') ]['VARIABLE'].values.tolist()
            print('feature_num',feature_num)
            pairs=list(itertools.combinations(feature_num,2))
            df_corr= pd.DataFrame()
            i=0
            for pair in pairs:
                i=i+1
                #print(i)
                if i%10000 == 0:
                        print(i)
                df= pd.DataFrame({pair[0]: X_train[pair[0]], pair[1]: X_train[pair[1]]})
                
                count_row= df.index.size
                count_any_na= df.isnull().values.any(axis=1).sum()
                count_all_na= df.isnull().values.any(axis=1).sum()

                if count_any_na -count_all_na < count_row * size_coeff:
                        df = df.dropna()
                        corr = np.corrcoef( df[pair[0]], df[pair[1]])[0,1]
                        if abs(corr) > cut_off_corr:
                                df_corr.loc[str(pair),'VAR1']= pair[0]
                                df_corr.loc[str(pair),'VAR2']= pair[1]
                                df_corr.loc[str(pair),'CORR_COEFF']= corr
                                df_corr.loc[str(pair),'VAR1-UNIGINI']= unigini[unigini['VARIABLE']==pair[0]]['TRAIN_GINI'].values
                                df_corr.loc[str(pair),'VAR2-UNIGINI']= unigini[unigini['VARIABLE']==pair[1]]['TRAIN_GINI'].values
                                df_corr.loc[str(pair),'STATUS-DROP']= np.where(df_corr.loc[str(pair),'VAR1-UNIGINI']> df_corr.loc[str(pair),'VAR2-UNIGINI'],pair[1],pair[0])
            if df_corr.empty== False:
                df_corr.sort_values('CORR_COEFF', ascending=[False], inplace=True)
                df_corr.reset_index(drop=True,inplace=True)
                drop_list=[]
                dff= df_corr.copy()
                while dff.index.size>0:
                        drop_list.append(dff.loc[0,'STATUS-DROP'].item((0)))
                        dff=dff[(dff[['VAR1','VAR2']] !=dff.loc[0,'STATUS-DROP'].item((0))).all(1)]
                        dff.reset_index(inplace=True,drop=True)
                meta['STATUS']=np.where(meta['VARIABLE'].isin(drop_list),'DROP_CORR COEFF',meta['STATUS'])
            return meta,df_corr