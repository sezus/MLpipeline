#pip install cx_Oracle
import cx_Oracle as orcl
#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils.utils_p import YamlParser, Config_Paths, Database_Config, Import_Dataset_Config,Upload_Download_Pickle


class Import_Data_Into_OracleDB():
    """Import_Data_Into_OracleDB"""
    def __init__(self):
        """init"""
        self.yaml_file = YamlParser()
        self.config = self.yaml_file.get_yaml_file()

        # Input & Output Paths:
        self.config_paths = Config_Paths()
        self.raw_path = self.config_paths.get_raw_path()
        self.parent_path = self.config_paths.get_parent_path()
        # Data_Config
        self.import_conf =Import_Dataset_Config()
        self.input_files=self.import_conf.get_inputFiles()
        self.tags=self.import_conf.get_Tags()

        # Database_Configs
        self.database_config = Database_Config()
        self.p_username = self.database_config.get_p_username()
        self.p_password = self.database_config.get_p_password()
        self.p_service = self.database_config.get_p_service()
        self.p_host = self.database_config.get_p_host()
        self.p_port = self.database_config.get_p_port()
    
        logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('Import_Data_Into_OracleDB')  # __name__=projectA.moduleB
    
    def upload_datasets(self):
        con = orcl.connect(user=p_username, password=p_password, dsn=p_host+"/"+p_service+":"+p_port)
        cur = con.cursor()
        for i in self.input_files:
            for tag in self.tags:
                dataf=Upload_Download_Pickle().download_pickle_file(self.raw_path,i + tag)
                insert_db(dataf,statement,cur,conn)
        

        
def insert_db(df,statement,cur,con):    
    for i in df:
        print(i.shape)
        for a in range(i.shape[0]):
            values=i.iloc[a].values.astype(str)
            values="','".join(str(x) for x in list(values))
            statement =statement+values+ """')"""
            cur.execute(statement)
            con.commit()


def create_table(statement,cur,con):    
        #print(statement)
        #print(a)
        cur.execute(statement)
        con.commit()



