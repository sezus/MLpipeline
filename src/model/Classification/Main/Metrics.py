import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score,average_precision_score, make_scorer, confusion_matrix, classification_report, cohen_kappa_score
from src.utils.utils_p import Upload_Download_Pickle, Model_Configs, Config_Paths, Logging_Config
# Classification_Metrics
class Metrics():
        def __init__(self):
                self.config = Config_Paths()
                self.processed_path = self.config.get_processed_path()
                self.model_path="model_path"
                self.model_name="model_name"
                #Load test values 
                self.y_tests_list='y_tests_list'
                self.y_test='y_test'
                self.y_pred_list="pred"
                self.average="av"
                self.metric_path = "metric"

                self.log_cfg=Logging_Config()        
                logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
                self.log='Classification'

        def Model_Metrics(self,type_,y_test=None,y_tests_list=None):

                #Load model, y_pred_list & y_pred
                y_pred,y_pred_proba,y_pred_train,y_pred_train_proba=self.load_prediction(self.model_name,type_)
                y_pred_list,y_pred_proba_list,y_pred_train_list,y_pred_train_proba_list=self.load_each_prediction(self.model_name,type_)

                self.logger = logging.getLogger(self.log)  # __name__=projectA.moduleB
                self.logger.debug('model_name %s' %self.model_name)
                self.logger.debug('y_pred_list %s :' %y_pred_list)
                if y_tests_list is None :
                        y_tests_list=self.load_each_test()
                        y_train_list=self.load_each_train()
                if y_test is None :
                        y_test=self.concat_data_set(y_tests_list)
                        y_train=self.concat_data_set(y_train_list)
                #Threshold analysis:
                
                thresholds=np.arange(0,1,0.01)
                metrics = dict.fromkeys(thresholds)
                print('y_pred_train_proba',y_pred_train_proba)
                df_y_pred_train_proba=pd.DataFrame(y_pred_train_proba)
                results_train = self.classification_model_metrics(metrics,y_train,df_y_pred_train_proba)
                results_train.to_excel('Train_Threshold.xlsx')

                perc_80=np.percentile(np.array(results_train['F1']),90)
                abs_min=min(results_train[results_train['F1']>perc_80]['Abs-diff'])
                df_y_pred_proba=pd.DataFrame(y_pred_proba)
                results_oot = self.classification_model_metrics(metrics,y_test,df_y_pred_proba)
                results_oot.to_excel('OOT_Threshold.xlsx')
                
                #selected threshold:
                selected_threshold=results_oot[(results_train['F1']>=perc_80) & (results_train['Abs-diff']>=abs_min)]
                
                #Lift
                #PSI
                
                #List scores
                self.List_scores_all(y_tests_list,y_pred_proba_list,threshold=selected_threshold,average=self.average)
                
                #DF scores
                self.DF_scores_all(y_test,y_pred_proba,threshold=selected_threshold,average=self.average)   


        def classification_model_metrics(self,metrics,y_act,y_pred):
                perf_metrics=['Accuracy','F1','Recall','Precision','Abs-diff','TP','FP','TN','FN','TP_FP','TP_FN','FPR','TPR','ROC','ROC_PA','APS']
                for idx,i in enumerate(metrics):
                        metrics[i]= perf_metrics.copy()
                        y_pred['PRED']= np.where(y_pred.iloc[:,1]>i,1,0)
                        metrics[i][0] = accuracy_score(y_act,y_pred['PRED'])
                        metrics[i][1] = f1_score(y_act,y_pred['PRED'])
                        metrics[i][2] = recall_score(y_act,y_pred['PRED'])
                        metrics[i][3] = precision_score(y_act,y_pred['PRED'])
                        metrics[i][4] = abs(recall_score(y_act,y_pred['PRED'])-recall_score(y_act,y_pred['PRED']))
                        
                        conf_mtx = confusion_matrix(y_act,y_pred['PRED'])
                        metrics[i][5] = conf_mtx[0][0]
                        metrics[i][6] = conf_mtx[0][1]
                        metrics[i][7] = conf_mtx[1][1]
                        metrics[i][8] = conf_mtx[1][0]
                        metrics[i][9] = conf_mtx[0][0] + conf_mtx[0][1]
                        metrics[i][10] = conf_mtx[0][0] + conf_mtx[1][0]
                        metrics[i][11] = conf_mtx[0][1] /( conf_mtx[0][1] + conf_mtx[0][0])
                        metrics[i][12] = conf_mtx[1][1] /( conf_mtx[1][1] + conf_mtx[1][0])
                        metrics[i][13] = roc_auc_score(y_act,y_pred['PRED'])
                        metrics[i][14] = roc_auc_score(y_act,y_pred[1])
                        metrics[i][15] = average_precision_score(y_act,y_pred['PRED'])
                metrics_df=pd.DataFrame(metrics).T
                metrics_df.columns=perf_metrics
                return metrics_df

        def concat_data_set(self, datalist):
                # for data in datalist:
                concatdata = pd.concat(datalist, ignore_index=True)
                return concatdata

        def load_test(self):
                y_test = Upload_Download_Pickle().download_pickle(self.processed_path, 'y_test')    
                return y_test

        def load_train(self):
                y_train = Upload_Download_Pickle().download_pickle(self.processed_path, 'y_train')    
                return y_train

        def load_each_test(self):
                y_test_list = Upload_Download_Pickle().download_pickle(self.processed_path, 'y_tests_list')  
                return y_test_list

        def load_each_train(self):
                y_trains_list = Upload_Download_Pickle().download_pickle(self.processed_path, 'y_trains_list')  
                return y_trains_list

        def load_prediction(self, model_name,type_=''):
                model_name_ = model_name +'_'+type_ + '_prediction.pckl'
                model_proba_name = model_name +'_'+type_ + '_proba_prediction.pckl'
                y_pred = Upload_Download_Pickle().download_pickle(self.model_path, model_name_)
                y_pred_proba = Upload_Download_Pickle().download_pickle(self.model_path, model_proba_name)

                model_train_name = model_name +'_'+type_ +'_train_prediction.pckl'
                model_train_proba_name = model_name +'_'+type_ + '_train_proba_prediction.pckl'
                print('model_train_name',model_train_name)
                print('model_train_proba_name',model_train_proba_name)
                y_pred_train = Upload_Download_Pickle().download_pickle(self.model_path, model_train_name)
                y_pred_train_proba = Upload_Download_Pickle().download_pickle(self.model_path, model_train_proba_name)

                return y_pred,y_pred_proba,y_pred_train,y_pred_train_proba

        def load_each_prediction(self, model_name,type_=''):
                model_name_ = model_name +'_'+type_ +'_each_prediction.pckl'
                model_proba_name = model_name +'_'+type_ + '_proba_each_prediction.pckl'
                print('model_proba_name',model_proba_name)
                y_pred = Upload_Download_Pickle().download_pickle(self.model_path, model_name_)
                y_pred_proba = Upload_Download_Pickle().download_pickle(self.model_path, model_proba_name)

                model_train_name_ = model_name +'_'+type_ +'train_each_prediction.pckl'
                model_train_proba_name = model_name +'_'+type_ + '_train_proba_each_prediction.pckl'
                y_pred_train = Upload_Download_Pickle().download_pickle(self.model_path, model_train_name_)
                y_pred_train_proba = Upload_Download_Pickle().download_pickle(self.model_path, model_train_proba_name)
                
                return y_pred,y_pred_proba,y_pred_train,y_pred_train_proba
      

        #def show_type_Metric(model_name,type_):
                #butun save edilen metricleri yukle ve dön!
        def show_model_Metric(self,model_name,type_):                
                #O modele ait bütün metricleri yukle ve dön
                metric_name = 'bests_' + model_name + type_ +'.pckl'
                metrics = Upload_Download_Pickle().download_pickle(self.model_path, metric_name)
                return metrics
                    
        def List_scores_all(self,y_tests_list,y_pred_proba_list,threshold=0.5,average='binary'):               
                # Optinal:Predict each testset seperately from
                y_pred_list=[]
                print('y_pred_proba_list[0]',y_pred_proba_list[0])
                
                for data in y_pred_proba_list:
                    data_=pd.DataFrame(data)
                    print('data : ',data_[1])
                    print('data 2: ',np.where(data_[1]>threshold,1,0))                    
                    y_pred_list.append(np.where(data_[1]>threshold,1,0))

                f1_test_list = self.f1_score_list(y_tests_list, y_pred_list,average)
                acc_score_list = self.acc_score_list(y_tests_list, y_pred_list,average)
                confusion_score_list = self.confusion_score_list(y_tests_list, y_pred_list,average)
                recall_score_list = self.recall_score_list(y_tests_list, y_pred_list,average)
                precision_score_list = self.precision_score_list(y_tests_list, y_pred_list,average)
                classification_report_list = self.classification_report_list(y_tests_list, y_pred_list,average)
                return f1_test_list,acc_score_list,confusion_score_list,recall_score_list,precision_score_list,classification_report_list

        def DF_scores_all(self,y_tests_list,y_pred_proba_list,threshold=0.5,average='binary'):
                y_pred_list=np.where(y_pred_proba_list[:,1]>threshold,1,0)
                f1_score_df = self.f1_score_df(y_tests_list, y_pred_list,average)
                acc_score_df = self.acc_score_df(y_tests_list, y_pred_list,average)
                confusion_score_df = self.confusion_score_df(y_tests_list, y_pred_list,average)
                recall_score_df = self.recall_score_df(y_tests_list, y_pred_list,average)
                precision_score_df = self.precision_score_df(y_tests_list, y_pred_list,average)
                classification_report_df = self.classification_report_df(y_tests_list, y_pred_list,average) 
                return f1_score_df,acc_score_df,confusion_score_df,recall_score_df,precision_score_df,classification_report_df      

        def save_score_list(self,type_,score):
                bests = {}    
                bests[type_ +'_seperated'] = score
                model_name = 'bests_' + self.model_name +'_'+  type_ + '_list.pckl'
                Upload_Download_Pickle().save_dataset_pickle(self.metric_path, model_name, bests)  

        def save_score_df(self,type_,score):
                bests = {}    
                bests[type_ +'_combined'] = score
                model_name = 'bests_' + self.model_name + '_'+ type_ +'_df.pckl'
                Upload_Download_Pickle().save_dataset_pickle(self.metric_path, model_name, bests)  

        def load_model(self,model_name,model_path):
                model_name = model_name + '_finalized_model.pckl'
                model = Upload_Download_Pickle().download_pickle(model_path, model_name)
                return model

        def f1_score_list(self,y_tests_list, y_pred_list, average='binary'):
                f1_test_list = []
                self.logger.warning('average %s :' %average)
                for i in range(len(y_pred_list)):
                        f1_test = f1_score(y_tests_list[i], y_pred_list[i],average=average)
                        # append each score to f1_test_list
                        f1_test_list.append(f1_test)
                self.save_score_list("f1_score",f1_test_list)
                self.logger.debug('f1_test_list %s :' %f1_test_list)
                return f1_test_list

        def f1_score_df(self,y_test, y_pred, average='binary'):
                f1_test = f1_score(y_test, y_pred,average=average)
                self.save_score_df("f1_score",f1_test)
                self.logger.debug('f1_score %s :' %f1_score)
                return f1_test
        # AUC/ROC
        # Accuracy_score
        def acc_score_list(self,y_tests_list, y_pred_list, average='binary'):
                accuracy_test_list = []
                for i in range(len(y_pred_list)):
                        accuracy_test = accuracy_score(y_tests_list[i], y_pred_list[i])
                        # append each score to f1_test_list
                        accuracy_test_list.append(accuracy_test)
                self.save_score_list("acc_score",accuracy_test_list)
                self.logger.debug('accuracy_test_list %s :' %accuracy_test_list)
                return accuracy_test_list

        def acc_score_df(self,y_test, y_pred, average='binary'):
                accuracy_test = accuracy_score(y_test, y_pred)
                self.save_score_df("acc_score",accuracy_test)    
                self.logger.debug('acc_score %s :' %accuracy_test)           
                return accuracy_test
        # Confusion_Matrix
        def confusion_score_list(self,y_tests_list, y_pred_list, average='binary'):
                confusion_list = []
                for i in range(len(y_pred_list)):
                        confusion = confusion_matrix(y_tests_list[i], y_pred_list[i])
                        # append each score to f1_test_list
                        confusion_list.append(confusion)
                self.save_score_list("confusion_score",confusion_list)
                self.logger.debug('confusion_score %s :' %confusion_list)
                return confusion_list

        def confusion_score_df(self,y_test, y_pred, average='binary'):
                confusion = confusion_matrix(y_test, y_pred)
                self.save_score_df("confusion_score",confusion)   
                self.logger.debug('confusion_df %s :' %confusion)               
                return confusion
        # Recall
        def recall_score_list(self,y_tests_list, y_pred_list, average='binary'):
                recall_test_list = []
                for i in range(len(y_pred_list)):
                        recall_test = recall_score(y_tests_list[i], y_pred_list[i],average=average)
                        # append each score to f1_test_list
                        recall_test_list.append(recall_test)
                self.save_score_list("recall_score",recall_test_list)
                self.logger.debug('recall_test_list %s :' %recall_test_list)      
                return recall_test_list

        def recall_score_df(self,y_test, y_pred, average='binary'):
                recall_test = recall_score(y_test, y_pred,average=average)
                self.save_score_df("recall_score",recall_test)    
                self.logger.debug('recall_test %s :' %recall_score)    
                return recall_test
        # Precision
        def precision_score_list(self,y_tests_list, y_pred_list, average='binary'):
                precision_test_list = []
                for i in range(len(y_pred_list)):
                        precision_test = precision_score(y_tests_list[i], y_pred_list[i],average=average)
                        # append each score to f1_test_list
                        precision_test_list.append(precision_test)
                self.save_score_list("precision_score",precision_test_list)
                self.logger.debug('precision_test_list %s :' %precision_test_list)   
                return precision_test_list

        def precision_score_df(self,y_test, y_pred, average='binary'):
                precision_test = precision_score(y_test, y_pred,average=average)
                self.save_score_df("precision_score",precision_test)  
                self.logger.debug('precision_test %s :' %precision_test)   
                return precision_test
        # Calculation
        def classification_report_list(self,y_tests_list, y_pred_list, average='binary'):
                classification_list = []
                for i in range(len(y_pred_list)):
                        classification = classification_report(y_tests_list[i], y_pred_list[i])
                        # append each score to f1_test_list
                        classification_list.append(classification)
                self.save_score_list("classification_score",classification_list)  
                self.logger.debug('classification_score %s :' %classification_list)                                     
                return classification_list

        def classification_report_df(self,y_test, y_pred, average='binary'):
                classification = classification_report(y_test, y_pred)
                self.save_score_df("classification_score",classification)   
                self.logger.debug('classification_score %s :' %classification)  
                return classification

        # Multi_Class_Classification_Metrics
        # F1 -- f1_micro f1_macro
        # AUC/ROC
        # Confusion_Matrix
        # Recall
        # Precision
        # Calculation
