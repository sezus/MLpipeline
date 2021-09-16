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
from src.featureProcessing.Imputation.Base import Imputation

class Basic_Imputation(Imputation):
    def __init__(self):
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
        self.log = 'Basic Imputation'  # __name__=projectA.moduleB

    def fill_data(self,dataf):
        print('fill_data')
        #print(self.fill_list)
        #birden fazla liste var ise:
        #print(self.imputation_conf_)
        if any(x == 'read_excel' for x in list(self.imputation_conf_.keys())) :
            read_excel=self.imputation_conf_['read_excel']
            if read_excel==False:                
                print('import_excel')
                self.imputation_fill_cnf=self.import_config()
                #print(self.imputation_fill_cnf.columns)
            else:
                print('import_data from yaml file fill list')
                #self.imputation_fill_cnf=self.import_config()
                excel_file=read_excel(self.combined_path+'\\Imputation_Excel.xlsx')
                # column_names=excel_file['Column_Name'].tolist()
                # lower_limit=excel_file['lowerlimit'].tolist()
                # upper_limit=excel_file['upperlimit'].tolist()
                # values=excel_file['value'].tolist()

        else:
            logging.raiseExceptions('Please add read_exce:False config to yaml file!!')
        
        for index,row in self.imputation_fill_cnf.iterrows():
            print('row_values',row.values)
            #self.imputation_fill_cnf=self.imputation_conf.get_list_conf(i)
            # 1.hangi kolonlar için imputation yapilacak:            
            feature_fill=row.column_name
            # feature_fill=column_names[i]
            # 2. limit koşulları neler
            upperlimit = row.upperlimit
            lowerlimit = row.lowerlimit
            #upperlimit=upper_limit[i]
            #lowerlimit=lower_limit[i]
            #print('feature_fill',feature_fill)
            #print('lowerlimit',lowerlimit)
            #print('upperlimit',upperlimit)  
            # 3. hangi değer ile değiştirilecek
            #value = self.imputation_fill_cnf[i]['values']
            value=row.value
            #print('value',value)
            if value!=np.nan:
                print('nan değil')
                # Eger value değeri verilmişse value değeriyle değiştirilecek.             
                # Remove Anomaly values from selected columns according to upper & lower limit
                #print(value) 
                dataf = self.remove_anomaly_values_single(dataf, feature_fill, int(upperlimit), int(lowerlimit),int(value))  
                #print('final_shape',dataf[ dataf['Blade2_act_val_A_degree']==-888888 ].shape)
            else:    
                # Eger value değeri verilmemişse np.nan ile değiştirilecek.
                # Remove Anomaly values from selected columns according to upper & lower limit
                dataf = self.remove_anomaly_values_single(dataf, feature_fill, upperlimit, lowerlimit)
                # Fill NA Empty Rows:
                # !!!eksik burda strategy koymak lazım. Ayrıca bunu sadece numeric type değişkenlere uygulamak lazım.
                dataf = self.fillna_withstrategy(dataf, 'ffill')
                dataf = self.fillna_withstrategy(dataf, 'bfill')                                
     
        return dataf

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
 
    def remove_anomaly_values_single(self, dataf, feature_fill_list,uplimit=None, lowlimit=None,value=np.nan ):
        #for  in feature_fill_list:
        print('remove_anomaly_values_single')
        #datafx=dataf[i]
        if uplimit != 'None':
            #print('uplimit', dataf[i])
            #self.logger.info('uplimit %s' % dataf.loc[dataf[i] > uplimit, i])
            #dataf.loc[dataf[i] > uplimit, i] = value,
            #print('value',value)
            #print('uplimit',uplimit)
            #print(dataf[feature_fill_list][dataf[feature_fill_list] > uplimit].shape)
            dataf[feature_fill_list]=np.where(dataf[feature_fill_list] > uplimit,value,dataf[feature_fill_list])
            #dataf.loc[dataf[i] > uplimit] = value
            print(dataf[dataf[feature_fill_list] > uplimit].shape)
        if lowlimit != 'None':
            #print('value',value)
            #print('lowlimit',lowlimit)
            #print('dataf', dataf[i])
            #self.logger.info('lowlimit %f' % dataf.loc[dataf[i] < lowlimit, i])
            #dataf.loc[dataf[i] < lowlimit, i] = value
            #print(dataf[feature_fill_list][dataf[feature_fill_list] < int(lowlimit)].shape)
            dataf[feature_fill_list]=np.where(dataf[feature_fill_list] < int(lowlimit),value,dataf[feature_fill_list])
            #dataf.loc[dataf[i] < int(lowlimit)] = value
        #print('dataf',dataf.shape)
        return dataf
    
    def remove_anomaly_values_withvalue(self, dataf, feature_fill_list,value, uplimit=None, lowlimit=None):
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

    def import_config(self):
        column_names=[]
        upper_limit=[]
        lower_limit=[]
        values=[]
        for i in self.fill_list:
            #print('list',i)
            self.imputation_fill_cnf=self.imputation_conf.get_list_conf(i)
            if any(x == 'feature_fill_list' for x in list(self.imputation_fill_cnf.keys())) & any(x == 'limit_conditions' for x in list(self.imputation_fill_cnf.keys())):
                feature_fill=self.imputation_fill_cnf['feature_fill_list']            
                for feature in feature_fill:
                    #print('feature',feature)
                    column_names.append(feature)
                    # 2. limit koşulları neler
                    upper_limit.append(self.imputation_fill_cnf['limit_conditions'][1])
                    lower_limit.append(self.imputation_fill_cnf['limit_conditions'][0])
                    #print('lowerlimit',self.imputation_fill_cnf['limit_conditions'][0])
                    #print('upperlimit',self.imputation_fill_cnf['limit_conditions'][1]) 
                    # 3. hangi değer ile değiştirilecek
                    if any(x == 'value' for x in list(self.imputation_fill_cnf.keys())):
                        values.append(self.imputation_fill_cnf['value'])  
                    else:
                        values.append(np.nan)  
            else:
                raise ValueError('feature_fill_list  or limit_conditions config is missing!')                 
        full_list=pd.DataFrame({'column_name':column_names,'lowerlimit':lower_limit,'upperlimit':upper_limit,'value':values})
        #return column_names,lower_limit,upper_limit,values
        return full_list

def fill_data_x(self,dataf):
        print('fill_data')
        #print(self.fill_list)
        #birden fazla liste var ise:
        if any(x == 'read_excel' for x in list(self.imputation_conf.keys())) :
            read_excel=self.imputation_fill_cnf['read_excel']
            if read_excel==False:                
                print('import_excel')
            else:
                print('import_data from yaml file fill list')
                self.fill_list=self.import_config()

        for i in self.fill_list:
            #print('list',i)
            #self.imputation_fill_cnf=self.imputation_conf.get_list_conf(i)
            # 1.hangi kolonlar için imputation yapilacak:            
            if any(x == 'feature_fill_list' for x in list(self.imputation_fill_cnf.keys())) & any(x == 'limit_conditions' for x in list(self.imputation_fill_cnf.keys())):
                feature_fill=self.imputation_fill_cnf['feature_fill_list']
                # 2. limit koşulları neler
                limit_cond=self.imputation_fill_cnf['limit_conditions']
                # 3. hangi değer ile değiştirilecek
                upperlimit = self.imputation_fill_cnf['limit_conditions'][1]
                lowerlimit = self.imputation_fill_cnf['limit_conditions'][0]
                #print('lowerlimit',lowerlimit)
                #print('upperlimit',upperlimit)            
                if any(x == 'value' for x in list(self.imputation_fill_cnf.keys())):
                    # Eger value değeri verilmişse value değeriyle değiştirilecek.             
                    # Remove Anomaly values from selected columns according to upper & lower limit
                    #print(self.imputation_fill_cnf['value'] )
                    value=self.imputation_fill_cnf['value']
                    dataf = self.remove_anomaly_values_single(dataf, feature_fill, upperlimit, lowerlimit,value)  
                    #print('final_shape',dataf[ dataf['Blade2_act_val_A_degree']==-888888 ].shape)
                else:    
                    # Eger value değeri verilmemişse np.nan ile değiştirilecek.
                    # Remove Anomaly values from selected columns according to upper & lower limit
                    dataf = self.remove_anomaly_values_single(dataf, feature_fill, upperlimit, lowerlimit)
                    # Fill NA Empty Rows:
                    # !!!eksik burda strategy koymak lazım. Ayrıca bunu sadece numeric type değişkenlere uygulamak lazım.
                    dataf = self.fillna_withstrategy(dataf, 'ffill')
                    dataf = self.fillna_withstrategy(dataf, 'bfill')                                
            else:
                raise ValueError('feature_fill_list  or limit_conditions config is missing!')          
        return dataf


