import os
import sys
sys.path.append('..') 
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from src.utils.utils_p import Upload_Download_Pickle, Config_Paths, Split_Train_Test_Configs, Logging_Config

class Split_Train_Test():
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

    def split_train_testset(self):
        # Process file path to save Splited Train, Test datasets       
        # Download Combined & Selected Datasets as List and DataFrames
        # 1. Input : DataFrames: Features_df,Labels_df & Lists: Features_List,Labels_List
        features_df,labels_df,features_list,labels_list=self.pickle_file.download_pickle_files(self.combined_path,self.input)
        self.logger.warning('Features_df %s' %features_df.columns)
        self.logger.warning('Labels_df %s' %labels_df.columns)
        #print('Features_df',Features_df.columns)
        #print('Labels_df',Labels_df.columns)
        
        # 2. Split : DataFrame split train test:
        # DataFrame
        # Drop unnecessary Columns to prepare ready datasets for training: 
        features_df=self.drop_columns_from_Dataframes(features_df,self.dropfeatures) 

        X_trains, X_test, y_trains, y_test = train_test_split(features_df,labels_df[self.target_column],test_size=0.25,random_state=30,shuffle=True)
        print('X_test',X_test.shape)
        print('y_test',y_test.shape)
        # 3. DataList     
        # Drop unnecessary Columns to prepare ready datasets for training:    
        features_list=self.drop_columns_from_List_Dataframes(features_list,self.dropfeatures)  

        X_trains_list,X_tests_list,y_trains_list,y_tests_list = self.traintestsplit(features_list,labels_list,0.2)

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

    def split_train_testset_Df(self,features_df,labels_df):
        # Process file path to save Splited Train, Test datasets       
        # 1. Input : DataFrames: Features_df,Labels_df
        self.logger.warning('Features_df %s' %features_df.columns)
        self.logger.warning('Labels_df %s' %labels_df.columns)        
        # 2. Split : DataFrame split train test:
        # DataFrame
        # Drop unnecessary Columns to prepare ready datasets for training: 
        features_df=self.drop_columns_from_Dataframes(features_df,self.dropfeatures) 
        X_trains, X_test, y_trains, y_test = train_test_split(features_df,labels_df[self.target_column],test_size=0.25,random_state=30,shuffle=True)
           
        #Save List and DataFrame train,test sets as :
        self.save_output_DataFrame_files(self.processed_path,X_trains,X_test,y_trains,y_test)

    def split_train_testset_List(self,features_list,labels_list):
        # Process file path to save Splited Train, Test datasets       
        # 3. DataList     
        # Drop unnecessary Columns to prepare ready datasets for training:    
        features_list=self.drop_columns_from_List_Dataframes(features_list,self.dropfeatures)        
        X_trains_list,X_tests_list,y_trains_list,y_tests_list = self.traintestsplit(features_list,labels_list,0.2)
        
        #Save List train,test sets as :
        self.save_output_List_files(self.processed_path,X_trains_list,X_tests_list,y_trains_list,y_tests_list)
           

    @staticmethod
    def upload_input_files(input,combined_path):
        pickle_list=[]
        for i in input:
            s=Upload_Download_Pickle().download_pickle(combined_path,i)
            pickle_list.append(s)
        return tuple(pickle_list)

    @staticmethod
    def save_output_DataFrame_files(path,X_trains,X_test,y_trains,y_test):
        #Save Feature,Label DataFrames:    
        Upload_Download_Pickle().save_dataset_pickle(path,'X_trains',X_trains)
        Upload_Download_Pickle().save_dataset_pickle(path,'X_test',X_test)    
        Upload_Download_Pickle().save_dataset_pickle(path,'y_trains',y_trains)
        Upload_Download_Pickle().save_dataset_pickle(path,'y_test',y_test) 

    @staticmethod       
    def save_output_List_files(path,X_trains_list,X_tests_list,y_trains_list,y_tests_list):   
        #List: X_trains_list,X_tests_list,y_trains_list,y_tests_list        
        Upload_Download_Pickle().save_dataset_pickle(path,'X_trains_list',X_trains_list)
        Upload_Download_Pickle().save_dataset_pickle(path,'X_tests_list',X_tests_list)    
        Upload_Download_Pickle().save_dataset_pickle(path,'y_trains_list',y_trains_list)
        Upload_Download_Pickle().save_dataset_pickle(path,'y_tests_list',y_tests_list) 

    @staticmethod
    def drop_columns_from_Dataframes(DF,refcolumn):
        i_df=DF.copy()
        i_df=i_df.drop(refcolumn,axis='columns')
        #print(i_df.columns)
        return i_df

    @staticmethod
    def drop_columns_from_List_Dataframes(List,refcolumn):
        List_df=[]
        for i in List:
            i_df=i.copy()
            i_df=i_df.drop(refcolumn,axis='columns')
            #print(i_df.columns)
            List_df.append(i_df) 
        return List_df  

    @staticmethod
    def convert_dataframe_to_list(df,refcolumn):
        #Convert dataframe to list according to reference column unique values
        X_list=[]
        for i in df[refcolumn].unique():
            ref_df=df[df[refcolumn]==i].copy()
            ref_df.drop(refcolumn,axis='columns')
            X_list.append(ref_df) 
        return X_list

    @staticmethod
    def traintestsplit(X_list, y_list, test_size_val):
        #türbin datalarını birleştirmek üzere kullanılır.     
        #list içerisinde verilmiş birden fazla feature ve label set'ini verilen test oranına göre ayrıştırır.
        X_trains=[]
        X_tests=[]
        X_oots=[]
        y_trains=[]
        y_tests=[]
        y_oots=[]
        for i in range(len(X_list)):
            print('X_list[i] shape:',X_list[i].shape)
            print('y_list[i] shape:',y_list[i].shape)
            if X_list[i].shape[0]!=0:
                X_train,X_test,y_train,y_test = train_test_split(X_list[i],y_list[i],test_size=test_size_val,shuffle=True)
                X_trains.append(X_train)
                X_tests.append(X_test)
                y_trains.append(y_train)
                y_tests.append(y_test)
        return X_trains,X_tests,y_trains,y_tests
    @staticmethod
    def ootsplit(X_list, y_list, test_size_val):
        #türbin datalarını birleştirmek üzere kullanılır.     
        #list içerisinde verilmiş birden fazla feature ve label set'ini verilen datelere göre ayrıştırır.
        X_trains=[]
        X_tests=[]
        y_trains=[]
        y_tests=[]
        for i in range(len(X_list)):
            print('X_list[i] shape:',X_list[i].shape)
            print('y_list[i] shape:',y_list[i].shape)
            if X_list[i].shape[0]!=0:
                X_train,X_test,y_train,y_test = train_test_split(X_list[i],y_list[i],test_size=test_size_val,shuffle=True)
                X_trains.append(X_train)
                X_tests.append(X_test)
                y_trains.append(y_train)
                y_tests.append(y_test)
        return X_trains,X_tests,y_trains,y_tests    
    