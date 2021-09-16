import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
sys.path.append('../..')
from src.model.Classification.Main.Metrics import Metrics
from src.model.Classification.Main.Metric_Plots import Metric_Plots
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from src.utils.utils_p import Upload_Download_Pickle, Model_Configs, Config_Paths, Logging_Config


class Classifier(Metrics,Metric_Plots):

    def __init__(self):

        self.config = Config_Paths()
        self.processed_path = self.config.get_processed_path()
        self.model='model'
        self.model_config = Model_Configs()
        self.input=self.model_config.get_inputFiles()
        self.model_path = self.model_config.get_models_path()
        self.model_name = 'Classifier'
        self.parameters = False
        self.number_of_cv_folds=5
        self.grid_search_=False

        self.results='results'
        self.param1='oaram1'
        self.param2='param2'

        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.log='Classification'
        #self.logger = logging.getLogger('Classification')  # __name__=projectA.moduleB
        self.pickle_file=Upload_Download_Pickle()   


    def getName(self):
        return self.model_name

    def __str__(self):
        return "Classifier is a %s" %(self.model_name)

    def Model(self):
        """ Default Classifier Model With Classification yaml config parameters:
            Perform the following steps for simple Classification:
            *Load X_trains_list, y_trains_list, X_tests_list, y_tests_list
            *Create a whole DataFrame from list of datasets
            *Create Model from predefined parameters and configs
            *Save Model
            *Predict & Score Model according to test values
            *Save results      
        """

        # Download Train Dataframes for Feature Selection:
        # X_trains, y_trains, X_test, y_test = self.download_train_test_Df(processed_path)
        self.logger = logging.getLogger(self.log)  # __name__=projectA.moduleB
        self.logger.warning('self.input %s' %self.input)
        # Download Train List for Feature Selection:
        X_trains_list, y_trains_list, X_tests_list, y_tests_list, self.selectedfeatures = self.pickle_file.download_pickle_files(self.processed_path,self.input)

        # List input convert to Dataframe to train or use Dataframes:
        X_trains, y_trains, X_tests, y_test = self.concat_List(
            X_trains_list, y_trains_list, X_tests_list, y_tests_list)

        # Fit Model
        fitmodel = self.fit_model(X_trains=X_trains, y_trains=y_trains,model=self.model,parameters=self.parameters,number_of_cv_folds=self.number_of_cv_folds,grid_search_=self.grid_search_)
        
        # Save Model
        self.save_Model(fitmodel, self.model_name,'classification')
        self.logger.warning('save_model')

        # Optional:Predict each testset seperately from 
        y_pred_list,y_pred,y_pred_list_proba,y_pred_proba=self.predict_(X_tests_list,X_tests,self.model_name,'classification')
        y_pred_train_list,y_pred_train,y_pred_train_list_proba,y_pred_train_proba=self.predict_(X_trains_list,X_trains,self.model_name,'classification')
        
        #save train prediction for whole dataset and each turbine
        #whole dataset
        self.save_prediction(y_pred_train,self.model_name,'classification_train')
        self.save_prediction(y_pred_train_proba,self.model_name,'classification_train_proba')

        #Each turbine
        self.save_each_prediction(y_pred_train_list,self.model_name,'classification_train')
        self.save_each_prediction(y_pred_train_list_proba,self.model_name,'classification_train_proba')

        #save oot prediction for whole dataset and each turbine
        #whole dataset
        self.save_prediction(y_pred,self.model_name,'classification')
        self.save_prediction(y_pred_proba,self.model_name,'classification_proba')

        #Each turbine
        self.save_each_prediction(y_pred_list,self.model_name,'classification')
        self.save_each_prediction(y_pred_list_proba,self.model_name,'classification_proba')
        #Calculate roc score:

        #Calculate f1 score:
        #Predict each testset
        f1_test_list = self.score_List(y_pred_list, y_tests_list,y_pred_list_proba)
        self.logger.warning('f1_test_list %s' %f1_test_list)

        # Predict whole testset
        f1_test = self.score_DF(y_pred, y_test,y_pred_proba)
        self.logger.warning('f1_test %s' %f1_test)
        # Define Dictionary

        self.save_results(f1_test_list, f1_test, self.model_name,'classification')
    
    def Multi_Model(self):
        """ Default Classifier Model With Classification yaml config parameters:
            Perform the following steps for simple Classification:
            *Load X_trains_list, y_trains_list, X_tests_list, y_tests_list
            *Create a whole DataFrame from list of datasets
            *Create Model from predefined parameters and configs
            *Save Model
            *Predict & Score Model according to test values
            *Save results      
        """

        # Download Train Dataframes for Feature Selection:
        # X_trains, y_trains, X_test, y_test = self.download_train_test_Df(processed_path)
        print('self.input',self.input)
        self.logger = logging.getLogger(self.log)  # __name__=projectA.moduleB
        self.logger.warning('self.input %s' %self.input)
        # Download Train List for Feature Selection:
        X_trains_list, y_trains_list, X_tests_list, y_tests_list, self.selectedfeatures = self.pickle_file.download_pickle_files(self.processed_path,self.input)
        print('X_trains_list',X_trains_list.shape)
        print('y_trains_list',y_trains_list.shape)
        print('X_tests_list',X_tests_list.shape)
        print('y_tests_list',y_tests_list.shape)
        # List input convert to Dataframe to train or use Dataframes:
        X_trains, y_trains, X_tests, y_test = self.concat_List(
            X_trains_list, y_trains_list, X_tests_list, y_tests_list)

        # Fit Model
        fitmodel = self.fit_model(X_trains=X_trains, y_trains=y_trains,model=self.model,parameters=self.parameters,number_of_cv_folds=self.number_of_cv_folds,grid_search_=self.grid_search_,scoring='f1_micro')
        # Save Model
        self.save_Model(fitmodel, self.model_name,'multi_classification')
        self.logger.warning('save_model')
        
        # Optinal:Predict each testset seperately from 
        y_pred_list,y_pred,y_pred_list_proba,y_pred_proba=self.predict_(X_tests_list,X_tests,self.model_name,'multi_classification')

        #save prediction for whole dataset and each turbine
        self.save_prediction(y_pred,self.model_name,'multi_classification')
        self.save_each_prediction(y_pred_list,self.model_name,'multi_classification')

        #Calculate f1 score:
        #Predict each testset
        print('self.average',self.average)
        f1_test_list = self.score_List(y_pred_list, y_tests_list,self.average)
        self.logger.warning('f1_test_list %s' %f1_test_list)

        # Predict whole testset
        print('self.average',self.average)
        f1_test = self.score_DF(y_pred, y_test,self.average)
        self.logger.warning('f1_test %s' %f1_test)
        # Define Dictionary

        self.save_results(f1_test_list, f1_test, self.model_name,'multi_classification')

        # Optinal:Predict each testset seperately from X_tests_list
        #f1_test_list = self.score_List(X_tests_list, y_tests_list, self.model_name, average='micro')
        #self.logger.warning('f1_test_list %s' %f1_test_list)

        # Predict whole testset
        #f1_test = self.score_DF(X_tests, y_test, self.model_name, average='micro')
        #self.logger.warning('f1_test %s' %f1_test)
        # Define Dictionary

        #self.save_results(f1_test_list, f1_test, self.model_name)    

    def fit_model(self, X_trains, y_trains, model, parameters,grid_search_=False,number_of_cv_folds=5,scoring=make_scorer(f1_score)):  
        """ fit_model
            Parameters
            ----------   
            X_trains
            y_trains
            model
            parameters
            grid_search_
            number_of_cv_folds
            score
        """      
        selectedfeatures = Upload_Download_Pickle().download_pickle(self.processed_path, 'selectedfeatures')
        print('selected_features   :   ',selectedfeatures)
        print('X_trains[selectedfeatures]  :   ',X_trains[selectedfeatures].head())
        # Create GridSearchClassifier
        if grid_search_ == True:
            model = self.grid_search(X=X_trains[selectedfeatures], y=y_trains, classifier=model, parameters=parameters, number_of_cv_folds=number_of_cv_folds, scoring=scoring)
        else:
            model.fit(X_trains[selectedfeatures], y_trains)
        return model

    def grid_search(self, X, y, classifier, parameters, number_of_cv_folds, scoring=make_scorer(f1_score)):
        """ fit_model
            Parameters
            ----------   
            X_trains
            y_trains
            classifier/model
            parameters
            number_of_cv_folds
            scoring
        """      
        gridsearcher = GridSearchCV(
            classifier, parameters, cv=number_of_cv_folds, scoring=scoring)
        print(gridsearcher.get_params())
        # Train with Training set X_Trains,y_trains
        gridsearcher.fit(X, y)
        return gridsearcher

    def Predict(self, X_tests_list,X_test,type_):
        """ score_List
            Parameters
            ----------   
            X_tests_list
            y_tests_list
            model_name
        """   
        # selectedfeatures:
        selectedfeatures = Upload_Download_Pickle().download_pickle(self.processed_path, 'selectedfeatures')
        # load model:
        gridsearcher = self.load_model(self.model_name,type_)
        y_pred_list,y_pred_proba_list=self.predict_list( X_tests_list,selectedfeatures,gridsearcher)
        y_pred,y_pred_proba= self.predict_DF(X_test,selectedfeatures,gridsearcher)
        #save prediction for whole dataset and each turbine
        self.save_prediction(y_pred,self.model_name,type_)
        self.save_prediction(y_pred_proba,self.model_name+'_proba',type_)
        self.save_each_prediction(y_pred_list,self.model_name,type_)
        self.save_each_prediction(y_pred_proba_list,self.model_name+'_proba',type_)
        
    def predict_(self, X_tests_list,X_test,model_name,type_):
        """ score_List
            Parameters
            ----------   
            X_tests_list
            y_tests_list
            model_name
        """   
        # selectedfeatures:
        selectedfeatures = Upload_Download_Pickle().download_pickle(self.processed_path, 'selectedfeatures')
        # load model:
        gridsearcher = self.load_model(model_name,type_)
        self.logger.warning('X_tests_list %s' %X_tests_list[0].columns)
        self.logger.warning('selectedfeatures %s' %selectedfeatures)
        y_pred_list,y_pred_proba_list=self.predict_list( X_tests_list,selectedfeatures,gridsearcher)
        y_pred,y_pred_proba= self.predict_DF(X_test,selectedfeatures,gridsearcher)
        return y_pred_list,y_pred,y_pred_proba_list,y_pred_proba

    def score_List(self, y_pred_list, y_tests_list,y_pred_list_proba,average='binary'):
        """ score_List
            Parameters
            ----------   
            X_tests_list
            y_tests_list
            model_name
        """   
        # score:
        f1_test_list=self.score_list( y_tests_list, y_pred_list,average)

        #f1_test_list = []
        #for i in range(len(X_tests_list)):
        #    y_pred = gridsearcher.predict(X_tests_list[i][selectedfeatures])
        #    f1_test = f1_score(y_tests_list[i], y_pred)
        #    # append each score to f1_test_list
        #    f1_test_list.append(f1_test)

        return f1_test_list

    def score_DF(self, y_pred,y_test,y_pred_proba, average='binary'):
    #def score_DF(self, X_test, y_test, model_name, average='binary'):
        #print(model_name)
        #selectedfeatures = Upload_Download_Pickle().download_pickle(self.processed_path, 'selectedfeatures')
        # Optinal:Predict each testset seperately from X_tests_list
        #gridsearcher = self.load_model(model_name)
        #y_pred=self.predict_DF(X_test,selectedfeatures,gridsearcher)
        f1_test=self.score_df(y_test,y_pred,average)
        #y_pred = gridsearcher.predict(X_test[selectedfeatures])
        #f1_test = f1_score(y_test, y_pred)
        return f1_test

    @staticmethod
    def predict_list( X_tests_list,selectedfeatures,gridsearcher):
        y_pred_list = []
        y_pred_proba_list =[]
        for i in range(len(X_tests_list)):
            print('X_tests_list',X_tests_list[i].columns)
            print('selectedfeatures',selectedfeatures)
            y_pred = gridsearcher.predict(X_tests_list[i][selectedfeatures])
            y_pred_proba = gridsearcher.predict_proba(X_tests_list[i][selectedfeatures])
            y_pred_list.append(y_pred)
            y_pred_proba_list.append(y_pred_proba)
        return y_pred_list,y_pred_proba_list

    @staticmethod
    def predict_DF(X_test,selectedfeatures,gridsearcher):
        y_pred = gridsearcher.predict(X_test[selectedfeatures])
        y_pred_proba = gridsearcher.predict_proba(X_test[selectedfeatures])
        return y_pred,y_pred_proba

    @staticmethod
    def score_list(y_tests_list, y_pred_list,average='binary'):
        f1_test_list = []
        for i in range(len(y_pred_list)):
            f1_test = f1_score(y_tests_list[i], y_pred_list[i],average=average)
            # append each score to f1_test_list
            f1_test_list.append(f1_test)
        return f1_test_list

    @staticmethod
    def score_df(y_test,y_pred,average='binary'):
        f1_test = f1_score(y_test, y_pred,average=average)
        return f1_test

    def load_model(self, model_name,type_=''):
        model_name = model_name  +'_'+type_ +'_finalized_model.pckl'
        model = Upload_Download_Pickle().download_pickle(self.model_path, model_name)
        return model

    def save_Model(self, gridsearcher, model_name,type_=''):
        model_name = model_name +'_'+type_ +'_finalized_model.pckl'
        Upload_Download_Pickle().save_dataset_pickle(self.model_path, model_name, gridsearcher)

    # def load_prediction(self, model_name,type_=''):
    #     model_name = model_name +'_'+type_ + '_prediction.pckl'
    #     y_pred = Upload_Download_Pickle().download_pickle(self.model_path, model_name)
    #     return y_pred
    # def load_each_prediction(self, model_name,type_=''):
    #     model_name = model_name +'_'+type_ +'_each_prediction.pckl'
    #     y_pred = Upload_Download_Pickle().download_pickle(self.model_path, model_name)
    #     return y_pred

    def save_prediction(self, y_pred, model_name,type_=''):
        model_name = model_name +'_'+type_ +'_prediction.pckl'
        Upload_Download_Pickle().save_dataset_pickle(self.model_path, model_name,y_pred )

    def save_each_prediction(self, y_pred, model_name,type_=''):
        model_name = model_name +'_'+type_ + '_each_prediction.pckl'
        Upload_Download_Pickle().save_dataset_pickle(self.model_path, model_name,y_pred )

    # def load_results(self, model_name,type_=''):
    #     model_name = 'bests_' + model_name +'_'+type_ +'.pckl'
    #     model = Upload_Download_Pickle().download_pickle(self.model_path, model_name)
    #     return model
        
    def save_results(self,f1_test_list,f1_test,model_name,type_=''):
        bests = {}    
        model=self.load_model(model_name,type_)
        print(self.grid_search_,'self.grid_search_')
        if self.grid_search_:
            bests['best_parameters'] = model.best_params_
            bests['cv_results'] = model.cv_results_
            bests['f1_test_seperated'] = f1_test_list
            bests['f1_test_combined'] = f1_test
        else:
            bests['f1_test_seperated'] = f1_test_list
            bests['f1_test_combined'] = f1_test
        print(bests)
        model_name = 'bests_' + model_name +'_'+type_ +'.pckl'
        Upload_Download_Pickle().save_dataset_pickle(self.model_path, model_name, bests)

    # def download_train_test_List(self):
    #     # Download Train List for Feature Selection:
    #     X_trains_list = Upload_Download_Pickle().download_pickle(self.processed_path, 'X_trains_list')
    #     y_trains_list = Upload_Download_Pickle().download_pickle(self.processed_path, 'y_trains_list')
    #     X_tests_list = Upload_Download_Pickle().download_pickle(self.processed_path, 'X_tests_list')
    #     y_tests_list = Upload_Download_Pickle().download_pickle(self.processed_path, 'y_tests_list')
    #     X_trains_list = Upload_Download_Pickle().download_pickle(self.processed_path, 'X_trains_list')

    #     return X_trains_list, y_trains_list, X_tests_list, y_tests_list

    # def download_train_test_Df(self):
    #     # Download Train Dataframes for Feature Selection:
    #     X_trains = Upload_Download_Pickle().download_pickle(self.processed_path, 'X_trains')
    #     y_trains = Upload_Download_Pickle().download_pickle(self.processed_path, 'y_trains')
    #     X_test = Upload_Download_Pickle().download_pickle(self.processed_path, 'X_test')
    #     y_test = Upload_Download_Pickle().download_pickle(self.processed_path, 'y_test')

    #     return X_trains, y_trains, X_test, y_test

    def concat_List(self, X_trains_list, y_trains_list, X_tests_list, y_tests_list):
        X_trains = self.concat_data_set(X_trains_list)
        y_trains = self.concat_data_set(y_trains_list)

        X_test = self.concat_data_set(X_tests_list)
        y_test = self.concat_data_set(y_tests_list)
        return X_trains, y_trains, X_test, y_test

    @staticmethod
    def concat_data_set( datalist):
        # for data in datalist:
        concatdata = pd.concat(datalist, ignore_index=True)
        return concatdata
    @staticmethod
    def model_train( X, y, classifier, parameters=None):
        classifier.fit(X, y)
        return classifier

    #def set_model_params(self):
    #    return self.parameters
