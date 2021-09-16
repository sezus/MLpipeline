import pickle
import numpy as np
import pandas as pd
import yaml
import os
import logging
import sys
sys.path.append('..')
from src.utils.utils_p import YamlParser, Config_Paths, Dataset_Configs, Upload_Download_Pickle, CombineDataset_Configs,Logging_Config



class Combine_Datasets():
    def __init__(self):
        # Input & Output Paths:
        self.config = Config_Paths()
        self.intermediate_path = self.config.get_intermediate_path()
        self.combined_path = self.config.get_combineddatasets_path()

        # Target Label Column & Identifier Tag Column
        self.target_column = self.config.get_Target_Column()
        self.tag_column = self.config.get_Tag_Column()

        # Combine_Dataset_Configs:
        self.combinedataconf = CombineDataset_Configs()
        self.datasets=self.combinedataconf.get_Upload_Datasets()

        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.logger = logging.getLogger('Combine_Datasets')  # __name__=projectA.moduleB

    def multi_combine_dataset(self):

        for data in self.datasets:
            self.upload_tag = self.combinedataconf.get_combine_Tags(data)
            self.input = self.combinedataconf.get_inputFiles(data)
            self.event_threshold = self.combinedataconf.get_Event_Threshold(data)
            self.output = self.combinedataconf.get_Output(data)
            self.combine_dataset(self.input,self.output,self.event_threshold)

    def combine_dataset(self,input,output,event_threshold):
        """1.Create A List of Datasets from Each Tag
           2.Create DataFrames from List  
           3.Optional: Only Select Datasets has enough Target Examples
        """
        # 1.Create A List of Datasets from Each Tag:
        features_list = self.combine_data(input[0])
        labels_list = self.combine_data(input[1])

        # 2.Create DataFrames from List:
        features_df = self.create_df_from_list(features_list)
        labels_df = self.create_df_from_list(labels_list)
        self.logger.debug('Combined_DataFrame_Feature_Columns : %s' % features_df.columns)     
        self.logger.debug('Combined_DataFrame_Label_Columns : %s' %labels_df.columns)    
        
        # 3.Optional: Only Select Datasets has enough Target Examples:
        self.filter_tag_target_list(features_list, labels_list,output,event_threshold)
        self.filter_tag_target_df(features_df, labels_df,output,event_threshold)

    def combine_data(self, arg):
        """This class does blah blah."""
        # 1.Create A List of Datasets from Each Tag:
        combine_list = []

        self.logger.warning('Combine Tags : %s' %self.upload_tag) 
        for tag in self.upload_tag:
            # Download pickle
            tag_dataf = Upload_Download_Pickle().download_pickle(
                self.intermediate_path, arg + tag)
            tag_dataf[self.tag_column] = tag
            combine_list.append(tag_dataf)
        #ortak kolonları al    
        common_column_list=combine_list[0].columns        
        for i in range(1,len(combine_list)):
            #common_column_list=np.intersect1d(combine_list[i].columns, combine_list[i+1].columns)
            common_column_list=common_column_list.intersection(combine_list[i].columns)
        #ortak kolonlara göre dataframe listesini yeniden oluştur
        combine_list_final=[]
        for dataset in combine_list:
            dataset=dataset[common_column_list]
            combine_list_final.append(dataset)

        return combine_list_final

    def create_df_from_list(self, arg):
        """This class does blah blah."""
        self.logger.warning('create_df_from_list : ') 
        arg_dataf = self.concat_data_set(arg)
        # Concat function put NA for all uncommon columns.
        # It suppose to set as '0' since fault is not occured.
        #Labels_df[Labels_df.isna()] == 0
        self.logger.warning('check na values : %s' %arg_dataf.isna().sum()) 
        #print('i', arg_dataf.isna().sum())
        arg_dataf = arg_dataf.fillna(0)
        self.logger.debug('na values filled as 0: %s' %arg_dataf.isna().sum()) 
        return arg_dataf

    def filter_tag_target_list(self, Feature_List, Labels_List,output,event_threshold):
        """This class does blah blah."""
        # 2.Optional: Only Select Datasets has enough Target Examples:
        # Create DataFrames from List:
        self.logger.warning('filter_tag_target_list : ') 
        Labels_df = self.create_df_from_list(Labels_List)
        selectedTags,turbine_table = self.select_WT_fault_count_bigger_than_threshold(
            Labels_df, self.tag_column, self.target_column, event_threshold)
        self.logger.warning('select_WT_fault_count_bigger_than_threshold : %s ' % selectedTags) 
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path,'turbine_table_list',turbine_table)
        turbine_table.to_excel('Turbine_Table_list.xlsx')
        # 3.Filter DataLists
        # Filter DataLists according to SelectedTags:
        Features_selected_list = self.filter_List_Tags(
            selectedTags, Feature_List)
        Labels_all_list = self.filter_List_Tags(selectedTags, Labels_List)

        # Filter other Labels then TargetColumn
        Labels_selected_list = self.filter_List_TargetColumn(
            self.target_column, Labels_all_list)

        # Save Feature,Label DataFrames:
        # DataLists
        self.save_Features_Labels_List(Features_selected_list,Labels_selected_list,Labels_all_list,output)
        #Upload_Download_Pickle().save_dataset_pickle(
        #    self.combined_path, 'Features_List', Features_selected_list)
        #Upload_Download_Pickle().save_dataset_pickle(
        #    self.combined_path, 'Labels_List', Labels_selected_list)
        #Upload_Download_Pickle().save_dataset_pickle(
        #    self.combined_path, 'AllLabels_List', Labels_all_list)

    def save_Features_Labels_List(self, Features_List,Labels_List,AllLabels_List,output):
        self.logger.debug('Save Features')
        print(output)
        Upload_Download_Pickle().save_dataset_pickle(
            self.combined_path,  output[0]  + '_List', Features_List)
        Upload_Download_Pickle().save_dataset_pickle(
            self.combined_path,  output[1] + '_List', Labels_List)
        Upload_Download_Pickle().save_dataset_pickle(
            self.combined_path,  output[2]  + '_List', AllLabels_List)

    def filter_tag_target_df(self, Features_df, Labels_df,output,event_threshold):
        """This class does blah blah."""
        self.logger.warning('filter_tag_target_df : ') 
        # Optional: Only Select Datasets has enough Target Examples:
        selectedTags,turbine_table = self.select_WT_fault_count_bigger_than_threshold(
            Labels_df, self.tag_column, self.target_column, event_threshold)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path,'turbine_table_df',turbine_table)
        turbine_table.to_excel('Turbine_Table_df.xlsx')
        # Filter Dataframes according to SelectedTags:
        Features_selected, Labels_selectedTag = self.filter_df(
            selectedTags, Features_df, Labels_df)

        # Filter other Labels then TargetColumn
        Labels_selected = Labels_selectedTag[['Period',
            self.target_column, self.tag_column]]

        # Save Feature,Label DataFrames:
        self.save_Features_Labels_DF( Features_selected, Labels_selected, Labels_selectedTag, output)
        #Upload_Download_Pickle().save_dataset_pickle(
        #    self.combined_path, 'Features_df', Features_selected)
        #Upload_Download_Pickle().save_dataset_pickle(
        #    self.combined_path, 'Labels_df', Labels_selected)
        #Upload_Download_Pickle().save_dataset_pickle(
        #    self.combined_path, 'AllLabels_df', Labels_selectedTag)

    def save_Features_Labels_DF(self, Features,Labels,AllLabels,output):
        self.logger.debug('Save Features')
        Upload_Download_Pickle().save_dataset_pickle(
            self.combined_path,  output[0]  + '_df', Features)
        Upload_Download_Pickle().save_dataset_pickle(
            self.combined_path,  output[1] + '_df', Labels)
        Upload_Download_Pickle().save_dataset_pickle(
            self.combined_path,  output[2]  + '_df', AllLabels)

    def filter_df(self, selectedTags, *args):
        """This class does blah blah."""
        arg_l = []
        for arg in args:
            # Filter Dataframes according to SelectedTags:
            arg_selected = arg[arg[self.tag_column].isin(selectedTags)]
            arg_l.append(arg_selected)
        logging.debug(tuple(arg_l))
        return tuple(arg_l)

    def filter_List_Tags(self, selectedTags, arg):
        """filter_List_Tags"""
        selected_list = []
        for i in arg:
            select = i[i[self.tag_column].isin(selectedTags)].copy()
            if select.shape[0] != 0:
                # select=select[select.columns[1:-2]]
                selected_list.append(select)
                self.logger.warning('select.columns %s' %select.columns)
            else:
                print(i[self.tag_column].unique(), 'ait',
                      self.target_column, 'Kayıt yok.')
                self.logger.warning('%s ait %s kayit yok ' % (i[self.tag_column].unique(),self.target_column ))
        return selected_list

    def filter_List_TargetColumn(self, target_column, arg):
        """filter_List_TargetColumn"""
        selected_list = []
        for i in arg:
            if i.shape[0] != 0:
                self.logger.debug('select.columns %s' %i.columns)
                select = i[[target_column]]
                # select=select[select.columns[1:-2]]
                selected_list.append(select)
            else:
                print(i[self.tag_column].unique(), 'ait',
                      target_column, 'Kayıt yok.')
                self.logger.warning('%s ait %s kayit yok ' % (i[self.tag_column].unique(),self.target_column ))
        return selected_list

    @staticmethod
    def select_WT_fault_count_bigger_than_threshold(dataf, numbercolumn, labelcolumn, threshold):
        """select_WT_fault_count_bigger_than_threshold"""
        # count according to numberof reference column
        # (sum,mean,median ya da tanımlanan bir fonksiyon input olarak verilebilir)
        tagnumber = []
        labelcolumn_list=[]
        events_count=[]
        for i in dataf[numbercolumn].unique():
            # print(i)
            systemnumber = dataf[dataf[numbercolumn] == i].copy()
            print('Number of ',labelcolumn ,' Events :',systemnumber[labelcolumn].sum(),'on ', i )
            labelcolumn_list.append(i)
            events_count.append(systemnumber[labelcolumn].sum())
            if systemnumber[labelcolumn].sum() > threshold:
                tagnumber.append(i)
        Turbine_Table=pd.DataFrame({'Tag':labelcolumn_list,'Events':events_count})
        
        return tagnumber,Turbine_Table

    @staticmethod
    def concat_data_set( datalist):
        """concat_data_set"""
        # for data in datalist:

        concatdata = pd.concat(datalist, ignore_index=True)
        return concatdata

    @staticmethod
    def convert_dataframe_to_list(dataf, refcolumn):
        """convert_dataframe_to_list"""
        # Convert dataframe to list according to reference column unique values
        # türbin datalarını birleştirmek üzere kullanılır.
        X_list = []
        for i in dataf[refcolumn].unique():
            ref_dataf = dataf[dataf[refcolumn] == i].copy()
            ref_dataf.drop(refcolumn, axis='columns')
            X_list.append(ref_dataf)
        return X_list
