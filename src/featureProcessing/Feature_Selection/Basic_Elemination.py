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
from sklearn.feature_selection import VarianceThreshold
class Basic_Elemination(FeatureSelection):
    def __init__(self):
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.model_path=self.model_config.get_models_path()       
        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Basic_Elemination'  # __name__=projectA.moduleB
        self.precondition = 'All_train_data'

        # Feature Selection Configs
        self.feature_selection_conf=Feature_Selection_Configs()

        # Method config 
        self.feature_selection_method_conf =self.feature_selection_conf.get_FeatureSelection_method_conf()
        self.feature_selection_method_conf_details = self.feature_selection_method_conf[self.log]
      
        # General config
        self.input=self.feature_selection_conf.get_inputFiles()        
        self.many_unique_cut_off=self.feature_selection_method_conf_details['many_unique_cut_off']
        self.createmetadata=self.feature_selection_method_conf_details['Create_metadata']

        #metadata upload
        self.pickle_file=Upload_Download_Pickle()   
        self.metaconf=self.feature_selection_conf.get_metaFile()
        #print('self.metaconf: ',str(self.metaconf[0]))        
        self.meta=self.pickle_file.download_pickle(self.processed_path,'meta')
        #print('self.meta',self.meta)       

    def filter_feature_selection(self,X_trains, y_trains):
            metaconf=self.feature_selection_conf.get_metaFile()
            print('self.metaconf: ',str(metaconf[0]))
            meta=self.pickle_file.download_pickle(self.processed_path,'meta')
            print(meta)
            #self.meta=self.pickle_file.download_pickle(self.processed_path,str(self.metaconf[0]))

            meta_null=self.basic_elimination_null (X_trains, meta)
            meta_same=self.basic_elimination_same(X_trains,meta_null)  
            meta=self.variance_threshold_elemination(X_trains,meta_same) 
            selectedfeatures=meta[meta['STATUS']=='KEEP']['VARIABLE'].values        
            return selectedfeatures,meta

    def basic_elimination_null (self,X_trains,meta, default_cut_off=0.2, variable_column='VARIABLE', status_column='STATUS'):
            print('X_trains.shape :',X_trains.shape)
            if meta.index.name==variable_column:
                meta.reset_index(inplace=True)         
            print('meta',meta)
            meta_new=meta[meta[variable_column].isin(X_trains.columns)].copy()
            meta_new=meta_new[meta_new[status_column]=='KEEP'].copy()  
            features_num,features_char=meta_features(meta_new)
            basic=meta_new.copy()
            if basic.index.name!=variable_column:
                basic.set_index([variable_column],inplace=True)

            elimination_data=X_trains.copy()

            #hard coded yapmak yerine yaml dosyası ile verilecek hale getir.
            Default_values=[-999999,-888888]  
            elimination_data.replace(Default_values,np.nan) 

            basic_elimination=[]
            elimination_percentage=[]        
            basic_elem=elimination_data.apply(lambda x: x.count() + x.isnull().sum())
            basic_elem=pd.DataFrame(basic_elem)
            #print(basic_elem.columns)
            basic_elem.columns=['count']
            #print(basic_elem['count'].values)    
            basic_elem['number_of_defaults'] = elimination_data.apply(lambda x: x.isnull().sum()).values
            #print(basic_elem['number_of_defaults'].values) 
            not_zero=basic_elem[basic_elem['number_of_defaults']!=0]
            print('not_zero: ',not_zero.head())
            if not_zero.shape[0]!=0:  
                not_zero['default_percentage'] = not_zero['number_of_defaults']/not_zero['count']
                #basic_elem['default_percentage'] = np.where(basic_elem.index.isin(not_zero.index),not_zero['default_percentage'],0)
                not_zero.columns=['count','number_of_defaults','default_percentage']
                for i in not_zero.index:
                        defperct=not_zero[not_zero.index==i]['default_percentage'].values[0]
                        elimination_percentage.append(defperct)
                        print(defperct)
                        if defperct>default_cut_off:
                            basic_elimination.append(i)
                            basic.loc[i,status_column]='DROP_na'
            basic.reset_index(inplace=True)
            meta[status_column]=np.where(meta[variable_column].isin(basic[basic[status_column]!='KEEP'][variable_column]),'DROP_na',meta[status_column])
            return meta
        

    def basic_elimination_same(self,data,meta, variable_column='VARIABLE', status_column='STATUS'):   
            basic=pd.DataFrame()
            basic.index.name='VARIABLE'
            if meta.index.name==variable_column:
                print(variable_column)
                meta.reset_index(inplace=True)         
            print('meta_same:',meta)
            print('meta_same name:',meta.index.name)
            meta_new=meta[meta[variable_column].isin(data.columns)].copy()
            meta_new=meta_new[meta_new[status_column]=='KEEP'].copy()
            features_num,features_char=meta_features(meta_new)
            basic=meta_new.copy()
            if basic.index.name!=variable_column:
                basic.set_index([variable_column],inplace=True)
            for var in features_num:
                    print(var)
                    if data[[var]].iloc[:,0].nunique(dropna=False)==1:
                        basic.loc[var,status_column]='DROP_all_same'
                    elif data[[var]].iloc[:,0].isnull().values.all():
                        basic.loc[var,status_column]='DROP_all_na'


            for var in features_char:
                    print(var)
                    data[var] = np.where(data[var].isin([np.nan]) == True, 'NaN', data[var])
                    if data[[var]].iloc[:,0].nunique(dropna=False)==1:
                        basic.loc[var,status_column]='DROP_all_same'
                    elif data[[var]].iloc[:,0].isnull().values.all():
                        basic.loc[var,status_column]='DROP_all_na'
                    elif data[[var]].iloc[:,0].nunique(dropna=False)> self.many_unique_cut_off:
                        basic.loc[var,status_column]='DROP_many_unique'

            basic.reset_index(inplace=True)            
            meta[status_column]=np.where(meta[variable_column].isin(basic[basic[status_column]=='DROP_all_same'][variable_column]),'DROP_all_same',meta[status_column])
            meta[status_column]=np.where(meta[variable_column].isin(basic[basic[status_column]=='DROP_all_na'][variable_column]),'DROP_all_na',meta[status_column])
            meta[status_column]=np.where(meta[variable_column].isin(basic[basic[status_column]=='DROP_many_unique'][variable_column]),'DROP_many_unique',meta[status_column])
                        
            #basic_all_same = basic[basic['STATUS']=='DROP_all_same']['VARIABLE'].values
            #basic_all_na = basic[basic['STATUS']=='DROP_all_na']['VARIABLE'].values
            #basic_any_unique = basic[basic['STATUS']=='DROP_many_unique']['VARIABLE'].values
            return meta
    def variance_threshold_elemination(self,X_trains,meta,selected_threshold=0.001, variable_column='VARIABLE', status_column='STATUS'):
            print('variance_threshold_elemination:')
            if meta.index.name==variable_column:
                meta.reset_index(inplace=True)     
            print('meta variance:',meta)    
            meta_new=meta[meta[variable_column].isin(X_trains.columns)].copy()
            meta_new=meta_new[meta_new[status_column]=='KEEP'].copy()
            features_num,features_char=meta_features(meta_new)
            print('features_char',features_char)
            basic=meta_new.copy() 
           
            #output (143, 59)
            columns=X_trains.columns[X_trains.isna().sum()==0]
            columns_new=basic[(basic['TYPE']!='CHAR')&(basic[variable_column].isin(columns))]

            if columns_new.index.name!=variable_column:
                columns_new.set_index([variable_column],inplace=True)

            print('X_trains.shape:',X_trains.shape) 
            X_train_new=X_trains[columns_new.index]
            if X_train_new.shape[0]!=0:
                normalized_df = self.normalize_all(X_train_new,features_num )
                print('X_train_new.shape:',X_train_new.shape)                 
                var_filter = VarianceThreshold(threshold = selected_threshold)  
                train = var_filter.fit_transform(normalized_df)                
                #to get the count of features that are not constant
                print(train.shape )   
                # output (143, 56)
                print(len(normalized_df.columns[var_filter.get_support()]))  
                normalized_df.var().to_excel('Variance_results.xlsx')
                #output 56
            else:
                print('no new feature to KEEP!!')
            basic.reset_index(inplace=True)
            basic[status_column]=np.where(basic[variable_column].isin(X_train_new.columns[~var_filter.get_support()]),'DROP_variance_smaller',basic['STATUS'])
            meta[status_column]=np.where(meta[variable_column].isin(basic[basic[status_column]!='KEEP'][variable_column]),'DROP_variance_smaller',meta[status_column])
            return meta

    def normalize(self, val):
        val = (val-np.min(val))/(np.max(val)-np.min(val))
        return val

    def normalize_all(self, dataf, feature_list):
        datafn = dataf.copy()
        for i in feature_list:
            datafn[i] = self.normalize(datafn[i].values)
        return datafn

def meta_features(meta):
    features_num=meta[meta['TYPE']=='NUMERIC']['VARIABLE']
    features_char=meta[meta['TYPE']=='CHAR']['VARIABLE']
    return features_num,features_char 
