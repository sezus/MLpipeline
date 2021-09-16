import os
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from src.utils.utils_p import YamlParser, Config_Paths, Upload_Download_Pickle, Dataset_Configs, Logging_Config


class Clean_Rename():
    def __init__(self):
        self.config = Config_Paths()
        self.raw_path = self.config.get_raw_path()
        self.intermediate_path = self.config.get_intermediate_path()
        self.processed_path = self.config.get_processed_path()
        self.combined_path = self.config.get_combineddatasets_path()

        self.dataset_config = Dataset_Configs()
        #self.target_column = self.config.get_Target_Column()
        #self.tag_column = self.config.get_Tag_Column()
        self.upload_tag = self.dataset_config.get_Upload_Tags('Datasets')

        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.logger = logging.getLogger('Clean_Normalize')  

    def single_clean_rename(self, dataset, tagname):
        # Download Upload Files
        Logs = Upload_Download_Pickle().download_pickle(self.raw_path, 'Logs'+tagname)
        #####02_Clean & Normalize Logs(Features) data#####
        # 02_1.rename columns,normalize,fill empty cells, burda normalize ve fill empty cells adımları ayrılabilir.
        Logs = self.clean_rename(dataset, Logs)
        # Save Normalize Logs & Not Normalize Logs & Combined Dataset Logs+Faults:
        Upload_Download_Pickle().save_dataset_pickle(
            self.intermediate_path, 'Logs_normal_fill' + tagname, Logs)
        #Upload_Download_Pickle().save_dataset_pickle(self.intermediate_path,
        #                                             'Logs_not_normal_fill' + tagname, Logs_not_normal)
        #Upload_Download_Pickle().save_dataset_pickle(
        #    self.intermediate_path, 'Logs_all_fill' + tagname, Logs_all)

    def clean_rename(self, dataset, logs):
        #####02_Clean & Normalize Logs(Features) data#####
        # Dataset Information:
        if len(dataset['renamecolumns']) != 0:
            print(logs.index.name)
            if logs.index.name==dataset['refcolumn']:
               logs.reset_index(inplace=True)
            renamecolumns = dataset['renamecolumns']
            # 02_1.rename columns,normalize,fill empty cells, burda normalize ve fill empty cells adımları ayrılabilir.
            logs = logs.rename(columns=renamecolumns)
        # Remove anomaly values and fill. Normalize dataset
        #logs = self.anomaly_normalize(logs, dataset)
        #logs_all=pd.concat([logs.set_index(dataset['refcolumn']),logs_not_normal.set_index(dataset['refcolumn'])],axis=1)
        return logs

    def anomaly_normalize(self, dataf, dataset):
        # Fill anomaly values
        #if dataset['feature_fill'] != None:
        #    feature_fill = dataset['feature_fill']
            #dataf = self.anomaly(dataf, feature_fill)
        #else:
        #    raise ValueError('feature_fill config is missing!')

        # Copy dataf_before_normalize
        dataf_before_normalize = dataf.copy()

        # normalize all dataset
        #if dataset['normalize_feature_list'] != None:
        #    normalize_feature_list = dataset['normalize_feature_list']
        # Normalize all continious columns burda sadece verilen list ile verilen kolonları normalize eder
        #    dataf = self.normalize_all(dataf, normalize_feature_list)
        #else:
        #    raise ValueError('normalize_feature_list config is missing!')

        return dataf_before_normalize

    def anomaly(self, dataf, feature_fill):

        # Remove Anomaly values from selected columns according to upper & lower limit
        dataf = self.remove_anomaly_lists(dataf, feature_fill)
        # Fill NA Empty Rows:
        # !!!eksik burda strategy koymak lazım. Ayrıca bunu sadece numeric type değişkenlere uygulamak lazım.
        dataf = self.fillna_withstrategy(dataf, 'ffill')
        dataf = self.fillna_withstrategy(dataf, 'bfill')
        return dataf

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

    def fillna(self, dataf):
        features = pd.DataFrame(dataf.isnull().sum(), columns=list('A'))
        for i in list(range(0, features.size)):
            # if features.iat[i,0] is not '0':
            dataf[features.index[i]] = dataf[features.index[i]].fillna(
                method='ffill')
            dataf[features.index[i]] = dataf[features.index[i]].fillna(
                method='bfill')
        return(dataf)

    def fillna_withstrategy(self, dataf, strategy):
        features = pd.DataFrame(dataf.isnull().sum(), columns=list('A'))
        for i in list(range(0, features.size)):
            # if features.iat[i,0] is not '0':
            dataf[features.index[i]] = dataf[features.index[i]].fillna(
                method=strategy)
        return(dataf)

    # def normalize(self, val):
    #     val = (val-np.min(val))/(np.max(val)-np.min(val))
    #     return val

    # def normalize_all(self, dataf, feature_list):
    #     datafn = dataf.copy()
    #     for i in feature_list:
    #         datafn[i] = self.normalize(datafn[i].values)
    #     return datafn

    def combine(self, conf, tagname):
        logs = Upload_Download_Pickle().download_pickle(
            self.intermediate_path, 'Logs_normal_fill'+tagname)
        logs_not_normal = Upload_Download_Pickle().download_pickle(
            self.intermediate_path, 'Logs_not_normal_fill'+tagname)
        faults = Upload_Download_Pickle().download_pickle(self.raw_path, 'Faults'+tagname)

        # Logs=Upload_Download_Pickle().download_pickle(self.raw_path,conf['combine_datasets'][0]+TagName)
        # Faults=Upload_Download_Pickle().download_pickle(self.raw_path,conf['combine_datasets'][1]+TagName)

        self.logger.warning('logs %s tagname %s  :' % (logs.columns,tagname))
        self.logger.warning('faults %s tagname %s  :' % (faults.columns,tagname))
        self.logger.warning('logs_not_normal %s tagname %s  :' % (tagname,logs_not_normal.columns))

        # 02_2.Combine Logs and Fault(Target) dataframes according to Period,Start Time,End Time
        logs_faults = self.combine_datasets(logs, faults)
        logs_not_normal_faults = self.combine_datasets(logs_not_normal, faults)
        Upload_Download_Pickle().save_dataset_pickle(
            self.intermediate_path, 'Logs_Faults' + tagname, logs_faults)
        Upload_Download_Pickle().save_dataset_pickle(
            self.intermediate_path, 'Log_not_normal_Faults' + tagname, logs_not_normal_faults)

    def combine_datasets(self, logs, faults):
        # 02_2.Combine Logs and Fault(Target) dataframes according to Period,Start Time,End Time
        logs_faults = self.combine_dataf_Period(logs, ['Period', 'Events'], faults, [
                                                'Start Time', 'End Time', 'Event ID'])
        return logs_faults

    def combine_dataf_Period(self, dataf1, refcolumn1, dataf2, refcolumn2):
        dataf1[refcolumn1[1]] = 0
        for i in range(dataf2[refcolumn2].shape[0]):
            # dataf1'deki dosyasında Period degeri(refcolumn1[0]) ile
            # F1:refcolumn1[0] degerine göre filtrele:
            startT02 = dataf1.loc[dataf1[refcolumn1[0]]
                                  > dataf2[refcolumn2[0]][i]]
            # print(startT02)
            # F2:refcolumn2[1]'e göre tekrar filtrele:
            # ind=startT02[startT02[refcolumn1[0]]<dataf2[refcolumn2[1]][i]].index

            # Ayıklanan indexe göre refcolumn1[1] 'Events' kolonuyla refcolumn2[1] kolonunu eşleştir.
            dataf1.loc[startT02[startT02[refcolumn1[0]] < dataf2[refcolumn2[1]]
                                [i]].index, refcolumn1[1]] = dataf2[refcolumn2[2]][i]
        self.logger.info('Combine Faults & Label According to Events : %s' %dataf1.Events.unique())
        print('unique events :',dataf1.Events.unique())
        return dataf1
