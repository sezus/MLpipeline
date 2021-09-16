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
from src.featureProcessing.SplitDataset import Split_Train_Test

class Horizontal_split(Split_Train_Test):
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
        self.type='horizontal'