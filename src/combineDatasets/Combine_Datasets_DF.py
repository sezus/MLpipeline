import os
import sys

sys.path.append('..')
import yaml
import pandas as pd
import numpy as np
import pickle
from src.utils.utils_p import YamlParser, Config_Paths, Dataset_Configs, Upload_Download_Pickle,CombineDataset_Configs


class Combine_Datasets():
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

        # Create A List of Datasets from Each Tag:
        Feature_List=self.combine_data_List('Features')
        Labels_List=self.combine_data_List('Labels')

        # Create DataFrames from List:
        Features_df=self.create_DF_from_list(Feature_List)
        Labels_df=self.create_DF_from_list(Labels_List)
                
        # 2. Filter Datasets according to Event_Threshold and Labels according to Target Column
        self.filter_Tag_Target(Features_df,Labels_df)

 
    def filter_Tag_Target(self,Features_df,Labels_df):
        # Optional: Only Select Datasets has enough Target Examples:
        selectedTags = self.select_WT_fault_count_bigger_than_threshold(Labels_df, self.Tag_Column, self.Target_Column, self.Event_Threshold)

        # Filter Dataframes according to SelectedTags:
        Features_selected,Labels_selectedTag=self.filter_df(selectedTags,Features_df,Labels_df)

        # Filter other Labels then TargetColumn 
        Labels_selected = Labels_selectedTag[[self.Target_Column, self.Tag_Column]]

        # Save Feature,Label DataFrames:
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path, 'Features_df', Features_selected)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path, 'Labels_df', Labels_selected)
        Upload_Download_Pickle().save_dataset_pickle(self.combined_path, 'AllLabels_df', Labels_selectedTag)

    def combine_data_List(self,arg):
        # 1.Create A List of Datasets from Each Tag:
        combine_list = []
        for tag in self.Upload_tag:
            # Download pickle files
            tag_df = Upload_Download_Pickle().download_pickle(self.intermediate_path, arg + tag)
            tag_df[self.Tag_Column] = tag
            # Attach all datasets to List
            combine_list.append(tag_df)
        return  combine_list

    def create_DF_from_list(self,arg):          
        arg_df = self.concat_data_set(arg)
        # Concat function put NA for all uncommon columns.
        # It suppose to set as '0' since fault is not occured.
        #Labels_df[Labels_df.isna()] == 0
        print('i',arg_df.isna().sum())
        arg_df=arg_df.fillna(0)
        return arg_df

    def filter_df(self,selectedTags,*args):
        arg_l=[]
        for arg in args:
            # Filter Dataframes according to SelectedTags:
            arg_selected=arg[arg[self.Tag_Column].isin(selectedTags)]
            arg_l.append(arg_selected)
        print(tuple(arg_l))
        return tuple(arg_l)

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