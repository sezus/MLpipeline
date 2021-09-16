import os
import sys
sys.path.append('..\..') 
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, make_scorer
from src.utils.utils_p import YamlParser,Config_Paths,Upload_Download_Pickle, Model_Configs, Feature_Selection_Configs, Logging_Config

class FeatureSelection():
    def __init__(self):
        # Input & Output Path
        self.model_config = Model_Configs()
        self.config = Config_Paths()
        self.processed_path=self.config.get_processed_path()
        self.model_path=self.model_config.get_models_path()

        # Feature Selection Configs
        self.feature_selection_conf=Feature_Selection_Configs()
        self.input=self.feature_selection_conf.get_inputFiles()
        self.threshold = self.feature_selection_conf.get_Feature_Selection_Threshold()
        self.methods = [chi2, f_classif]
        self.method_names = ['chi2','f_classif']
        self.model = RandomForestClassifier
        self.method=f_classif
        self.classification=self.feature_selection_conf.get_Classification()
        self.average=self.feature_selection_conf.get_Average()
        self.rand=42
        self.pickle_file=Upload_Download_Pickle()       
        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log = 'Feature_Selection'  # __name__=projectA.moduleB
        self.precondition = 'Both_train_valid'
        self.metaconf=self.feature_selection_conf.get_metaFile()
        print('self.metaconf: ',str(self.metaconf[0]))
        self.meta=self.pickle_file.download_pickle(self.processed_path,'meta')
        print('self.meta',self.meta)

    def featureSelection(self):
        self.logger = logging.getLogger(self.log)  # __name__=projectA.moduleB
        # Download Train Dataframes for Feature Selection:
        #X_trains=Upload_Download_Pickle().download_pickle(self.processed_path,'X_trains')
        #y_trains=Upload_Download_Pickle().download_pickle(self.processed_path,'y_trains')
        # Download Train List for Feature Selection:
        #X_trains_list=download_pickle(processed_path,'X_trains_list')
        #y_trains_list=download_pickle(processed_path,'y_trains_list')
        X_trains,y_trains,X_trains_list,y_trains_list=self.pickle_file.download_pickle_files(self.processed_path,self.input)
        
        #Create meta data         
        if isinstance(self.meta, pd.DataFrame):
            print('metadata already created.')  
            print(self.createmetadata) 
            if self.createmetadata==True:
                print('Create new metadata and save:')
                self.meta=self.create_meta_data(X_trains)
                Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'meta',self.meta)
                             
        else:            
            print('metadata :',self.meta)
            print('Create new metadata and save:')
            self.meta=self.create_meta_data(X_trains)
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'meta',self.meta)
        
        #Split Train DataFrames to train and validation sets to validate results:
        print('X_trains',X_trains.shape)
        print('y_trains',y_trains.shape)
        X_train, X_valid, y_train, y_valid = train_test_split(X_trains,y_trains,test_size=0.2,random_state=42,stratify=y_trains)
        print('X_train',X_train.shape)
        print('X_valid',X_valid.shape)        
        #train ve validation ayırıp verilen feature selection algoritmaları için performans skorlaması yapıyoruz.
        #f1_train_all,f1_valid_all,column_names=self.filter_feature_selection(X_train, X_valid, y_train, y_valid,self.method,self.model,42,self.threshold)
        if self.precondition=='All_train_data':            
            selectedfeatures,meta=self.filter_feature_selection(X_trains, y_trains)
            self.logger.warning('selectedfeatures %s' %selectedfeatures)
        elif self.precondition=='Both_train_valid':
            selectedfeatures,meta=self.filter_feature_selection(X_train, X_valid, y_train, y_valid)
            self.logger.warning('selectedfeatures %s' %selectedfeatures)
        #mean_inc_threshold=sum(np.array(f1_train_all[1:30])-np.array(f1_train_all[0:29]))/len(np.array(f1_train_all[1:30])-np.array(f1_train_all[0:29]))
        #mean_step_threshold=mean_diff(f1_valid_all)
    
        #Sonuçlara göre thresholddan yuksek olanları seçiyoruz.
        #selectedfeatures=self.select_feature_performance_increase_bigger_than_threshold(f1_train_all,f1_valid_all,column_names,self.threshold)   
        print('final_meta',meta)
        Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'selectedfeatures',selectedfeatures)
        Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'meta',meta)
        

    def featureSelection_df(self,X_trains,y_trains):
        #Split Train DataFrames to train and validation sets to validate results:
        X_train, X_valid, y_train, y_valid = train_test_split(X_trains,y_trains,test_size=0.2,shuffle=True)
        
        #train ve validation ayırıp verilen feature selection algoritmaları için performans skorlaması yapıyoruz.
        #f1_train_all,f1_valid_all,column_names=self.filter_feature_selection(X_train, X_valid, y_train, y_valid,self.method,self.model,42,self.threshold)
        selectedfeatures=self.filter_feature_selection(X_train, X_valid, y_train, y_valid)
        #mean_inc_threshold=sum(np.array(f1_train_all[1:30])-np.array(f1_train_all[0:29]))/len(np.array(f1_train_all[1:30])-np.array(f1_train_all[0:29]))
        #mean_step_threshold=mean_diff(f1_valid_all)
    
        #Sonuçlara göre thresholddan yuksek olanları seçiyoruz.
        #selectedfeatures=self.select_feature_performance_increase_bigger_than_threshold(f1_train_all,f1_valid_all,column_names,self.threshold)   

        return selectedfeatures

    def featureSelection_list(self,X_trains,y_trains):
        #Split Train DataFrames to train and validation sets to validate results:
        X_train, X_valid, y_train, y_valid = train_test_split(X_trains,y_trains,test_size=0.2,shuffle=True)
        
        #train ve validation ayırıp verilen feature selection algoritmaları için performans skorlaması yapıyoruz.
        f1_train_all,f1_valid_all,column_names=self.filter_feature_selection(X_train, X_valid, y_train, y_valid)
        #mean_inc_threshold=sum(np.array(f1_train_all[1:30])-np.array(f1_train_all[0:29]))/len(np.array(f1_train_all[1:30])-np.array(f1_train_all[0:29]))
        #mean_step_threshold=mean_diff(f1_valid_all)
    
        #Sonuçlara göre thresholddan yuksek olanları seçiyoruz.
        selectedfeatures=self.select_feature_performance_increase_bigger_than_threshold(self.meta,f1_train_all,f1_valid_all,column_names,self.threshold)   

        return selectedfeatures

    def filter_feature_selections(self,X_train, X_valid, y_train, y_valid):
        method_names=self.method_names
        for i in range(len(self.method_names)):
            self.method=self.methods[i]            
            f1_train_all,f1_valid_all,column_names=self.filter_feature_selection(X_train, X_valid, y_train, y_valid)
            #Sonuçlara göre thresholddan yuksek olanları seçiyoruz
            selectedfeatures,self.meta=self.select_feature_performance_increase_bigger_than_threshold(self.meta,f1_train_all,f1_valid_all,column_names,0.005)   
            feature_score ={}
            feature_score['f1_train_all']=f1_train_all
            feature_score['f1_valid_all']=f1_valid_all
            feature_score['column_names']=column_names
            feature_score['selectedfeatures']=selectedfeatures
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,method_names[i],feature_score)
          
    def filter_feature_selection(self,X_train, X_valid, y_train, y_valid):
            f1_train_all=[]
            f1_valid_all=[]
            column_names=[]  
            print('self.meta',self.meta)          
            for k in range(1,X_train.shape[1]):
                columns=X_train.columns[X_train.isna().sum()==0]
                X_train_new=X_train[columns]
                Column_Names = self.filtering(X_train_new,y_train,self.method,k)
                #print('Column_Names',Column_Names)
                #print('Column_Names',X_train[X_train.columns[Column_Names]].head())
                f1_train, f1_valid = self.train_(X_train[X_train_new.columns[Column_Names]], y_train, X_valid[X_train_new.columns[Column_Names]], y_valid, self.model,self.rand,self.average)
                f1_train_all.append(f1_train)
                f1_valid_all.append(f1_valid)
                column_names.append(X_train.columns[Column_Names])           
                self.logger.warning('filter selection wrapper column_names %s with f_train %s and f_valid %s' %( X_train.columns[Column_Names], f1_train, f1_valid) )
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'f1_train_all',f1_train_all)
            Upload_Download_Pickle().save_dataset_pickle(self.processed_path,'f1_valid_all',f1_valid_all)
            selectedfeatures,meta=self.select_feature_performance_increase_bigger_than_threshold(self.meta,f1_train_all,f1_valid_all,column_names,self.threshold) 
            return selectedfeatures,meta

    @staticmethod
    def filtering(X,y,method,nf):
        Selector = SelectKBest(method, k=nf).fit(X, y)
        Column_Names = Selector.get_support(indices=True)
        return Column_Names

    @staticmethod
    def mean_diff(f1_valid_all):
        difflist=np.array(f1_valid_all[1:len(f1_valid_all)])-np.array(f1_valid_all[0:(len(f1_valid_all)-1)])
        step_threshold =sum(difflist)/len(difflist)
        return step_threshold
    
    #verilen train ve test setlerine ve modele göre f1_train ve f1_valid değerlerini döndürür
    @staticmethod
    def train_(X,y,X_t,y_t,model,rand,average='binary'):
        classifier = model(max_depth = 10, n_jobs = 10, random_state = rand)
        classifier.fit(X,y)
        y_pred_train = classifier.predict(X)
        y_pred = classifier.predict(X_t)
        f1_train = f1_score(y_pred_train, y,average=average)
        f1_valid = f1_score(y_pred, y_t,average=average)
        return f1_train, f1_valid

    #verilen train ve test setlerine ve modele göre f1_train ve f1_valid değerlerini döndürür
    @staticmethod
    def train_multiclass(X,y,X_t,y_t,model,rand,average='micro'):
        classifier = model(max_depth = 10, n_jobs = 10, random_state = rand)
        classifier.fit(X,y)
        y_pred_train = classifier.predict(X)
        y_pred = classifier.predict(X_t)
        f1_train = f1_score(y_pred_train, y, average=average)
        f1_valid = f1_score(y_pred, y_t, average=average)
        return f1_train, f1_valid
    
    ## count according to numberof reference column (sum,mean,median ya da tanımlanan bir fonksiyon input olarak verilebilir)
    @staticmethod
    def select_WT_fault_count_bigger_than_threshold(df,numbercolumn,labelcolumn,threshold):
        tagnumber=[]
        for i in df[numbercolumn].unique():
            #print(i)
            systemnumber=df[df[numbercolumn]==i].copy()
            #print(systemnumber[labelcolumn].sum())
            if systemnumber[labelcolumn].sum()>threshold:  
                tagnumber.append(i)
        return tagnumber

  
    def select_feature_performance_increase_bigger_than_threshold(self,meta,f1_train_all,f1_valid_all,column_names,step_threshold, variable_column='VARIABLE', status_column='STATUS'):
        #meta==meta[meta[variable_column].isin(data.columns)].copy()
        print('meta',meta)
        if meta.index.name==variable_column:
            meta.reset_index(inplace=True)  
        #meta_new=meta[meta[variable_column].isin(column_names)].copy()
        basic=meta[meta[status_column]=='KEEP'].copy()
        print('basic',basic)      
        #temp_f1 = f1_valid_all[0]
        #self.logger.warning('column_names' %column_names)
        temp_features = column_names[0]
        clist=f1_train_all[1:len(f1_train_all)]>np.array(f1_train_all[0:(len(f1_train_all)-1)])+step_threshold
        indexlist=np.where(clist)[0]+1 
        #self.logger.warning('indexlist' % indexlist)
        # hangi featureset kombinasyonlarında artış yaşanmış onların listesi indexlist. 
        # sadece artış olanları alıp önceki indexten farklı olarak eklenen featureları alıyoruz.
        for i in indexlist:
            #print('i',i)
            temp_features_new=column_names[i]
            #print('column_names:',column_names)
            print('index:',i)
            print('column_names[i]:',column_names[i])
            print('column_names[i-1]:',column_names[i-1])
            self.logger.warning('temp_features_new :  ' % temp_features_new)
            # diyelim 2,5,7 combinasyonları alıyoruz 5in 4'den farkı olarak eklenen feature'u alıyoruz
            temp_features_old=column_names[i-1]
            self.logger.warning('temp_features_old :  ' % temp_features_old)
            #Sonrasında eski featureları drop edip sadece 4'den 5'e geçiştekini alıyrouz.
            temp_features_new=temp_features_new.drop(temp_features_old)
            print('temp_features_new',temp_features_new.values[0])
            print('temp_features_new',temp_features_new.tolist())
            print('basic',basic)
            temp_features = temp_features.append(temp_features_new)
            print(basic)
        basic[status_column]=np.where(basic[variable_column].isin(temp_features),'KEEP','DROP_hybrid_threhold')
        meta[status_column]=np.where(meta[variable_column].isin(basic[basic[status_column]!='KEEP'][variable_column]),'DROP_hybrid_threhold',meta[status_column])
        return temp_features,meta

    @staticmethod    
    def create_meta_data(data):
        feature_type=np.where(data.dtypes=='float64', 'NUMERIC','CHAR')
        feature_type=np.where(data.dtypes=='int64', 'NUMERIC',feature_type)
        metadata={'VARIABLE': data.columns, 'TYPE':feature_type, 'STATUS':'KEEP'}
        meta=pd.DataFrame(metadata)
        return meta

    def meta_features(meta):
        features_num=meta[meta['TYPE']=='NUMERIC']['VARIABLE']
        features_char=meta[meta['TYPE']=='CHAR']['VARIABLE']
        return features_num,features_char    
    
    
    