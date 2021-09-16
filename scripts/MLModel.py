from abc import ABC, abstractmethod  
from src.utils.utils_p import YamlParser

class MLModel(ABC):
    '''An abstract base class for ML model prediction code '''
    def __init__(self):
        self.yaml_parser = YamlParser()

    @abstractmethod
    def Collect_Data(self):
        pass
    @abstractmethod
    def DataPreprocessing(self):
        pass
    @abstractmethod
    def CombineDataset(self):
        pass
    @abstractmethod
    def SplitTrainTest(self):
        pass
    @abstractmethod
    def FeatureSelection(self):
        pass
    @abstractmethod
    def fit_Model(self, data):
        pass
    @abstractmethod
    def predict_Model(self, data):
        pass