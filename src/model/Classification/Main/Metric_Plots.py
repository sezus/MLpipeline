import sys
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer, confusion_matrix, classification_report, cohen_kappa_score
from src.utils.utils_p import Upload_Download_Pickle, Model_Configs, Config_Paths, Logging_Config
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  

# Classification_Metrics
class Metric_Plots():
    def __init__(self):
        self.config = Config_Paths()
        self.processed_path = self.config.get_processed_path()
        self.model_path="model_path"
        self.model_name="model_name"
        self.y_tests_list="test"
        self.y_pred_list="pred"
        self.average="av"   
        self.results='results'
        self.param1='param1'
        self.param2='param2'

        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))

    def grid_plot(self):
        #grid results:
        results_=Upload_Download_Pickle().download_pickle(self.model_path,self.results)
        scores,param_1,param_2=self.grid_search_results(results_,[self.param1,self.param2])
        #logaritm:
        log_param2=[]
        for i in param_2:
            log_param2.append(np.log10(i))
        #plot
        self.plot_(scores,param_1,log_param2,self.model_name)


    @staticmethod
    def grid_search_results(best,params):
        scores=best['cv_results']['mean_test_score']
        print('scores',best['cv_results'])
        if best['cv_results'].get(params[0]) is None:
            param_1 = None
        else :
            param_1 = best['cv_results'][params[0]].data

        if best['cv_results'].get(params[1]) is None:
            param_2 = ''
        else :
            param_2 = best['cv_results'][params[1]].data

        return scores,param_1,param_2

    @staticmethod
    def plot_(scores,param_1,param_2, title):

        plt.style.use('classic')
        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(10)

        if param_1 == []:
            param_1= np.zeros(shape=len(param_2))
      
        if param_2 == []:
            param_2= np.zeros(shape=len(param_1))

        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(param_1, param_2, scores, marker = "^", s=30, c = scores)
        ax.scatter(param_1[scores.argmax()], param_2[scores.argmax()], max(scores), marker = "o", s=100, c = 'black')
        ax.set_xlabel('max_depth')
        ax.set_ylabel('min_child_weight(log scale)')
        ax.set_zlabel('mean validation f1 score')
        ax.view_init(30, 140)
        ax.set_title(title)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        text = ax.text(param_1[scores.argmax()], 
               param_2[scores.argmax()], 
               max(scores)+0.04, 
               'max_depth = '+str(param_1[scores.argmax()])+
              '\nmin_child_weight = '+str(param_2[scores.argmax()])+
              '\nmean validation score = '+str(max(scores)),
              [0,0,0], bbox=props)


    


