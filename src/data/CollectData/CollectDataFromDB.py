#libraries
import cx_Oracle as orcl
import pandas as pd
import numpy as np
import pickle
import os
import sys
import pandas as pd
import logging
from src.utils.utils_p import YamlParser, Config_Paths, Database_Config, Upload_Download_Pickle
sys.path.append('..')

class CollectDataFromDB():
    """CollectDataFromDB"""
    def __init__(self):
        """init"""
        self.yaml_file = YamlParser()
        self.config = self.yaml_file.get_yaml_file()

        # Input & Output Paths:
        self.config_paths = Config_Paths()
        self.raw_path = self.config_paths.get_raw_path()
        self.parent_path = self.config_paths.get_parent_path()

        # Dataset_Configs
        self.database_config = Database_Config()
        self.p_username = self.database_config.get_p_username()
        self.p_password = self.database_config.get_p_password()
        self.p_service = self.database_config.get_p_service()
        self.p_host = self.database_config.get_p_host()
        self.p_port = self.database_config.get_p_port()
    
        logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('CollectDataFromDB')  # __name__=projectA.moduleB

    def upload_all_datasets(self):
        con = orcl.connect(user=self.p_username, password=self.p_password, dsn=self.p_host+"/"+self.p_service+":"+self.p_port)
        cur = con.cursor()

        query_normalize = """SELECT* 
           FROM WINDTURBINE_NORMALIZE"""
        Windturbine_Normalize_Features = pd.read_sql(query_normalize, con=con)
        query_maincategorylabels = """SELECT* 
           FROM WINDTURBINE_MAINCATEGORYLABELS"""
        Windturbine_Components= pd.read_sql(query_maincategorylabels, con=con)
        query_raw = """SELECT* 
           FROM WINDTURBINE"""
        Windturbine= pd.read_sql(query_raw, con=con)

        with open('Windturbine_Components_', 'wb') as config_file:
            pickle.dump(Windturbine_Components, config_file)
        with open('Windturbine_Normalize_Features_', 'wb') as config_file:
            pickle.dump(Windturbine_Normalize_Features, config_file)
        with open('Windturbine_', 'wb') as config_file:
            pickle.dump(Windturbine, config_file)