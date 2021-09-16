import os
import sys
from numpy.core import overrides
sys.path.append('..') 
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from src.utils.utils_p import Upload_Download_Pickle, Config_Paths, Split_Train_Test_Configs, Logging_Config
from src.featureProcessing.SplitDataset.Split_Train_Test import Split_Train_Test

class Vertical_Split(Split_Train_Test):
    def __init__(self):
        # Input & Output Paths
        self.config = Config_Paths()
        ## Input path :
        self.combined_path=self.config.get_combineddatasets_path()
        ## Output path :
        self.processed_path=self.config.get_processed_path()

        # Target & Tag Column
        self.target_column=self.config.get_Target_Column()
        self.tag_column=self.config.get_Tag_Column()

        # Split_Train_Test_Configs
        self.split_conf=Split_Train_Test_Configs()
        self.input=self.split_conf.get_inputFiles()
        self.dropfeatures=self.split_conf.get_drop_Features()

        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.logger = logging.getLogger('Split_Train_Test')  # __name__=projectA.moduleB
        self.pickle_file=Upload_Download_Pickle()
        self.type=self.split_conf.get_type()
        self.reference_column=self.split_conf.get_referance_Column()
        self.selected_tag=self.split_conf.get_selected_data()


    def split_train_testset(self):
        print('Vertical')
        # Process file path to save Splited Train, Test datasets       
        # Download Combined & Selected Datasets as List and DataFrames
        # 1. Input : DataFrames: Features_df,Labels_df & Lists: Features_List,Labels_List
        features_df,labels_df,features_list,labels_list=self.pickle_file.download_pickle_files(self.combined_path,self.input)
        self.logger.warning('Features_df %s' %features_df.columns)
        self.logger.warning('Labels_df %s' %labels_df.columns)
        #print('Features_df',Features_df.columns)
        #print('Labels_df',Labels_df.columns)

        #selected_tags:
        print(self.selected_tag)

        X_trains=features_df[features_df[self.reference_column].isin(self.selected_tag)]
        y_trains=labels_df[labels_df[self.reference_column].isin(self.selected_tag)][self.target_column]     
        X_test=features_df[~features_df[self.reference_column].isin(self.selected_tag)]
        y_test=labels_df[~labels_df[self.reference_column].isin(self.selected_tag)][self.target_column]      
        
        # 2. Split : DataFrame split train test:
        # DataFrame
        # Drop unnecessary Columns to prepare ready datasets for training: 
        X_trains=self.drop_columns_from_Dataframes(X_trains,self.dropfeatures) 
        X_test=self.drop_columns_from_Dataframes(X_test,self.dropfeatures) 
                

        #X_trains, X_test, y_trains, y_test = train_test_split(features_df,labels_df[self.target_column],test_size=0.25,random_state=30,shuffle=True)
        print('X_test',X_test.shape)
        print('y_test',y_test.shape)
        # 3. DataList     
        # Drop unnecessary Columns to prepare ready datasets for training:    
   
        X_trains_list,X_tests_list,y_trains_list,y_tests_list = self.traintestsplit(features_list,labels_list)
        X_trains_list=self.drop_columns_from_List_Dataframes(X_trains_list,self.dropfeatures)  
        X_tests_list=self.drop_columns_from_List_Dataframes(X_tests_list,self.dropfeatures)  

        len_xtest=[]
        len_xtrain=[]
        for i in range(len(X_tests_list)):
            print('X_test_list',len(X_tests_list[i]))
            print('y_tests_list',len(y_tests_list[i]))
            len_xtest.append(len(y_tests_list[i]))

        for i in range(len(X_trains_list)):
            print('X_trains_list',len(X_trains_list[i]))
            print('y_trains_list',len(y_trains_list[i]))
            len_xtrain.append(len(y_trains_list[i]))

        #Save List and DataFrame train,test sets as :
        self.save_output_DataFrame_files(self.processed_path,X_trains,X_test,y_trains,y_test)
        self.save_output_List_files(self.processed_path,X_trains_list,X_tests_list,y_trains_list,y_tests_list)


    def traintestsplit(self,X_list, y_list):
        #türbin datalarını birleştirmek üzere kullanılır.     
        #list içerisinde verilmiş birden fazla feature ve label set'ini verilen test oranına göre ayrıştırır.
        X_trains=[]
        X_tests=[]
        y_trains=[]
        y_tests=[]
        for i in range(len(X_list)):
            print('X_list[i] shape:',X_list[i].shape)
            print('y_list[i] shape:',y_list[i].shape)
            selected_df=X_list[i]
            size_df=selected_df[selected_df[self.reference_column].isin(self.selected_tag)]
            if size_df.shape[0]!=0:
                X_trains.append(size_df)
                y_trains.append(y_list[i])
            else:
                X_tests.append(selected_df)
                y_tests.append(y_list[i])                
        return X_trains,X_tests,y_trains,y_tests

