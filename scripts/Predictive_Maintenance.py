import os
import sys
sys.path.append('..')
from src.MLModel import MLModel
from src.data.Collect_Data_From_CSV import Collect_Data_From_CSV
from src.intermediate.Data_Preprocessing import Data_Preprocessing
from src.intermediate.Combine_Datasets import Combine_Datasets
from src.featureProcessing.SplitDataset.Split_Train_Test import *
from src.featureProcessing.Feature_Selection.Wrapper import *
from src.featureProcessing.Feature_Selection.Hybrid import *
from src.model.Random_Forest_Tuning import Random_Forest_Tune
from src.model.XGBoost_Tuning import XGBoost_Tune
from src.model.LightGBM_Tuning import LightGBM_Tune
from src.model.CatBoost_Tuning import CatBoost_Tune


class PDModel(MLModel):
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
    
    def Collect_Data(self):
        self.CollectData.upload_all_datasets()

    def DataPreprocessing(self):
        self.DataPreprocessing.Preprocessing() 

    def CombineDataset(self):
        self.CombineDataset.combine_dataset() 

    def SplitTrainTest(self):
        self.SplitTrainTest.split_train_testset()

    def FeatureSelection(self):
        self.FeatureSelection.featureSelection()

    def fit_Model(self, data):
        self.CatBoostTune.Model()

    def predict_Model(self, data):
        self.CatBoostTune.Model()