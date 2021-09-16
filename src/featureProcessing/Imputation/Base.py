import sys
sys.path.append('..\..') 
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, make_scorer
from src.utils.utils_p import YamlParser,Config_Paths,Upload_Download_Pickle, Model_Configs, Imputation_Configs, Logging_Config

class Imputation():
    def __init__(self):
        # Input & Output Path
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.combined_path=self.config.get_combineddatasets_path()
        self.model_path=self.model_config.get_models_path()

        # Feature Selection Configs
        self.imputation_conf=Imputation_Configs()
        self.input=self.imputation_conf.get_inputFiles()
        self.imputation_conf_=self.imputation_conf.get_imputation_conf()
        self.fill_list=self.imputation_conf.get_fill_list()

        self.pickle_file=Upload_Download_Pickle()       
        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Imputation'  # __name__=projectA.moduleB
 
    def impute_before_split(self):
        print('fill_data')
        print(self.input)
        Features_df,Features_List=self.pickle_file.download_pickle_files(self.combined_path,self.input)
        print(Features_df.head())
        data_new=self.fill_data(Features_df)
        print('data_new',data_new.shape)
        data_list_new=[]
        for data in Features_List:
            print('data.shape',data.shape)
            data_list=self.fill_data(data)
            data_list_new.append(data_list)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path,  'Features_df_new', data_new)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path,  'Features_List_new', data_list_new)
        #return data_new

    def impute_after_split(self):
        X_test_list_new=[]
        X_trains_list_new=[]
        print('fill_data')
        #Normal datasets:
        print('Normal_datasets')
        X_trains,X_test,X_trains_list,X_test_list=self.pickle_file.download_pickle_files(self.processed_path,self.input)
        print(X_trains.head())
        X_trains_new=self.fill_data(X_trains)
        X_test_new=self.fill_data(X_test)
        #List datasets:
        print('List_datasets')
        for data in X_trains_list:
            #print('data.shape',data.shape)
            data_list=self.fill_data(data)
            X_trains_list_new.append(data_list)
        for data in X_test_list:
            #print('data.shape',data.shape)
            data_list=self.fill_data(data)
            X_test_list_new.append(data_list)           
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path,  'X_trains_new', X_trains_new)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path,  'X_test_new', X_test_new)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path,  'X_trains_list_new', X_trains_list_new)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path,  'X_test_list_new', X_test_list_new)
        return X_trains_new,X_test_new

    def fill_data(self,dataf):
        print('fill_data')
        #print(self.fill_list)
        for i in self.fill_list:
            self.imputation_fill_cnf=self.imputation_conf.get_list_conf(i)
            feature_fill=self.imputation_fill_cnf['feature_fill_list']
            self.imputation_fill_cnf['limit_conditions']
            self.imputation_fill_cnf['value']  
            
        if feature_fill!= None:
            dataf = self.anomaly(dataf, feature_fill)
        else:
            raise ValueError('feature_fill config is missing!')          

    def anomaly(self, dataf, feature_fill):

        # Remove Anomaly values from selected columns according to upper & lower limit
        dataf = self.remove_anomaly_lists(dataf, feature_fill)
        # Fill NA Empty Rows:
        # !!!eksik burda strategy koymak lazım. Ayrıca bunu sadece numeric type değişkenlere uygulamak lazım.
        dataf = self.fillna_withstrategy(dataf, 'ffill')
        dataf = self.fillna_withstrategy(dataf, 'bfill')
        return dataf

    def fillna_withstrategy(self, dataf, strategy):
        features = pd.DataFrame(dataf.isnull().sum(), columns=list('A'))
        for i in list(range(0, features.size)):
            # if features.iat[i,0] is not '0':
            dataf[features.index[i]] = dataf[features.index[i]].fillna(
                method=strategy)
        return(dataf)

    def remove_anomaly_lists(self, dataf, feature_fill):
        # get lists :list1,list2, ...listn
        for i in feature_fill['list']:
            # get list
            feature_fill_l = feature_fill[i]
            dataf = self.remove_anomaly_values(dataf, feature_fill_l)
        return dataf

    def remove_anomaly_values(self, dataf, feature_fill_l):
        if feature_fill_l['feature_fill_list'] != None and feature_fill_l['limit_conditions'] != None:
            feature_fill_list = feature_fill_l['feature_fill_list']
            # get upper and lower limit for those feature values:
            upperlimit = feature_fill_l['limit_conditions'][0]
            lowerlimit = feature_fill_l['limit_conditions'][1]
            dataf = self.remove_anomaly_values_single(
                dataf, feature_fill_list, upperlimit, lowerlimit)
        else:
            raise ValueError('feature_fill_l config is missing')
        return dataf

    def remove_anomaly_values_single(self, dataf, feature_fill_list, uplimit=None, lowlimit=None):
        for i in feature_fill_list:
            if uplimit != 'None':
                #print('uplimit', dataf[i])
                #self.logger.info('uplimit %s' % dataf.loc[dataf[i] > uplimit, i])
                dataf.loc[dataf[i] > uplimit, i] = np.nan
            if lowlimit != 'None':
                #print('uplimit', dataf[i])
                #self.logger.info('lowlimit %f' % dataf.loc[dataf[i] < lowlimit, i])
                dataf.loc[dataf[i] < lowlimit, i] = np.nan
        return dataf

