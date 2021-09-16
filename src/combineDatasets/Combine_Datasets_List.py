import os
import sys

sys.path.append('..')
import yaml
import pandas as pd
import numpy as np
import pickle
from src.utils.utils_p import YamlParser, Config_Paths, Dataset_Configs, Upload_Download_Pickle,CombineDataset_Configs


class Combine_Datasets_List():
    def __init__(self):
        # Input & Output Paths:
        self.config = Config_Paths()
        self.intermediate_path = self.config.get_intermediate_path()
        self.combined_path = self.config.get_combineddatasets_path()

        # Target Label Column & Identifier Tag Column
        self.Target_Column = self.config.get_Target_Column()
        self.Tag_Column = self.config.get_Tag_Column()

        # Combine_Dataset_Configs:
        self.combinedataconf=CombineDataset_Configs()
        self.Upload_tag=self.combinedataconf.get_combine_Tags()
        self.input=self.combinedataconf.get_inputFiles()
        self.Event_Threshold=200

    def combine_dataset(self):
        print(self.input)
        # 1.Create A List of Datasets from Each Tag:
        Feature_List=self.combine_data('Features')
        Labels_List=self.combine_data('Labels')
        
        # 2. Filter Datasets according to Event_Threshold and Labels according to Target Column
        self.filter_Tag_Target(Feature_List,Labels_List)

    def combine_data(self,arg):
        # 1.Create A List of Datasets from Each Tag:
        combine_list = []
        for tag in self.Upload_tag:
            # Download pickle files
            tag_df = Upload_Download_Pickle().download_pickle(self.intermediate_path, arg + tag)
            tag_df[self.Tag_Column] = tag
            # Attach all datasets to List
            combine_list.append(tag_df)
        return  combine_list

    def filter_Tag_Target(self,Feature_List,Labels_List):
        # 2.Optional: Only Select Datasets has enough Target Examples:
        # Create DataFrames from List:
        Labels_df=self.create_DF_from_list(Labels_List)
        selectedTags = self.select_WT_fault_count_bigger_than_threshold(Labels_df, self.Tag_Column, self.Target_Column, self.Event_Threshold)

        # 3.Filter DataLists
        # Filter DataLists according to SelectedTags:
        Features_selected_list=self.filter_List_Tags(selectedTags,Feature_List)
        Labels_all_list=self.filter_List_Tags(selectedTags,Labels_List)

        # Filter other Labels then TargetColumn 
        Labels_selected_list=self.filter_List_TargetColumn(self.Target_Column,Labels_all_list)

        # Save Feature,Label DataFrames:
        #DataLists
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path, 'Features_List', Features_selected_list)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path, 'Labels_List', Labels_selected_list)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path, 'AllLabels_List', Labels_all_list)

    def filter_List_Tags(self,selectedTags,arg):
        selected_list = []
        for i in arg:
            select = i[i[self.Tag_Column].isin(selectedTags)].copy()
            if select.shape[0] != 0:
                # select=select[select.columns[1:-2]]
                selected_list.append(select)
                print(select.columns)
            else:
                print(i[self.Tag_Column].unique(), 'ait', self.Target_Column, 'Kayıt yok.')
        return selected_list

    def filter_List_TargetColumn(self,TargetColumn,arg):
        selected_list = []
        for i in arg:
            if i.shape[0] != 0:
                print(i.columns)
                select = i[[self.Target_Column]]
                # select=select[select.columns[1:-2]]
                selected_list.append(select)
            else:
                print(i[self.Tag_Column].unique(), 'ait', self.Target_Column, 'Kayıt yok.')
        return selected_list

    def create_DF_from_list(self,arg):          
        arg_df = self.concat_data_set(arg)
        # Concat function put NA for all uncommon columns.
        # It suppose to set as '0' since fault is not occured.
        #Labels_df[Labels_df.isna()] == 0
        print('i',arg_df.isna().sum())
        arg_df=arg_df.fillna(0)
        return arg_df

    ## count according to numberof reference column (sum,mean,median ya da tanımlanan bir fonksiyon input olarak verilebilir)
    def select_WT_fault_count_bigger_than_threshold(self, df, numbercolumn, labelcolumn, threshold):
        tagnumber = []
        for i in df[numbercolumn].unique():
            # print(i)
            systemnumber = df[df[numbercolumn] == i].copy()
            # print(systemnumber[labelcolumn].sum())
            if systemnumber[labelcolumn].sum() > threshold:
                tagnumber.append(i)
        return tagnumber

    def concat_data_set(self, datalist):
        # for data in datalist:
        self.concatdata = pd.concat(datalist, ignore_index=True)
        return self.concatdata

    # Convert dataframe to list according to reference column unique values
    # türbin datalarını birleştirmek üzere kullanılır.
    def convert_dataframe_to_list(self, df, refcolumn):
        X_list = []
        for i in df[refcolumn].unique():
            ref_df = df[df[refcolumn] == i].copy()
            ref_df.drop(refcolumn, axis='columns')
            X_list.append(ref_df)
        return X_list
