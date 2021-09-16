import os
import sys
sys.path.append('..') 
from src.data.Collect_Data_From_CSV import Collect_Data_From_CSV
from src.intermediate.Data_Preprocessing import Data_Preprocessing
from src.intermediate.Combine_Datasets import Combine_Datasets
from src.processing.Split_Train_Test import Split_Train_test
from src.processing.Feature_Selection import FeatureSelection
from src.model.Random_Forest_Tuning import Random_Forest_Tune
from src.model.XGBoost_Tuning import XGBoost_Tune
from src.model.LightGBM_Tuning import LightGBM_Tune
from src.model.CatBoost_Tuning import CatBoost_Tune

class Machine_Learning: 
    '''Facade'''
  
    def __init__(self): 
        self.CollectData = Collect_Data_From_CSV()
        self.DataPreprocessing = Data_Preprocessing()
        self.CombineDataset = Combine_Datasets()
        self.SplitTrainTest=Split_Train_test()
        self.FeatureSelection=FeatureSelection()
        self.CatBoostTune=CatBoost_Tune()
        self.XGBoostTune=XGBoost_Tune()
        self.LightGBMTune=LightGBM_Tune()
        self.RandomForestTune=Random_Forest_Tune()
  
    def startML(self): 
        self.CollectData.upload_all_datasets()
        self.DataPreprocessing.Preprocessing()  
        self.CombineDataset.combine_dataset() 
        self.SplitTrainTest.split_train_testset()
        self.FeatureSelection.featureSelection()
        self.CatBoostTune.Model()
        self.XGBoostTune.Model()
        self.LightGBMTune.Model()
        self.RandomForestTune.Model()


""" main method """
if __name__ == "__main__": 
  
    ML = Machine_Learning() 
    ML.startML() 


  