import os
import yaml
import pandas as pd
import pickle
import re


class YamlParser:
    def __init__(self):
        # if itemlar null mı kontrol koy.
        import pathlib
        p = pathlib.Path().absolute()
        par = p.parent / 'src'
        par = os.path.join(p.parent, 'src')
        CONFIG_PATH = str(par)
        with open(os.path.join(CONFIG_PATH, 'importcsv.yaml'), 'r', encoding='utf8') as file:
            loader = yaml.SafeLoader
            loader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
                list(u'-+0123456789.'))
            self.yaml_file = yaml.load(file, Loader=loader)

    def get_yaml_file(self):
        return self.yaml_file

    def get_config_paths(self):
        return self.yaml_file["Upload_list_all"]


class Config_Paths:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.parent_path = self.yaml_conf['Parent_path']

    def get_parent_path(self):
        return self.parent_path

    def get_raw_path(self):
        self.raw_path = os.path.join(self.parent_path, 'data', '01_raw')
        return self.raw_path

    def get_intermediate_path(self):
        self.intermediate_path = os.path.join(
            self.parent_path, 'data', '02_preprocess')
        return self.intermediate_path

    def get_combineddatasets_path(self):
        self.combineddatasets_path = os.path.join(
            self.parent_path, 'data', '03_combinedDatasets')
        return self.combineddatasets_path

    def get_processed_path(self):
        self.processed_path = os.path.join(
            self.parent_path, 'data', '04_featureProcessing')
        return self.processed_path

    def get_model_path(self):
        self.model_path = os.path.join(self.parent_path, 'data', '05_models')
        return self.model_path

    def get_modelsOutputs_path(self):
        self.modelsOutputs_path = os.path.join(
            self.parent_path, 'data', '06_model_outputs')
        return self.modelsOutputs_path

    def get_Target_Column(self):
        self.Target_Column = self.yaml_conf["Target_Column"]
        return self.Target_Column

    def get_Tag_Column(self):
        self.Tag_Column = self.yaml_conf["Tag_Column"]
        return self.Tag_Column

    def get_Upload_Tags(self, config):
        self.Upload_Tags = config["Upload_Tags"]
        return self.Upload_Tags

    def get_Tag(self, TagName):
        self.Tag = self.yaml_conf[TagName]
        return self.Tag

    def get_Tag_Datasets(self, Datasets, TagName, DatasetName):
        self.Tag_Dataset = self.yaml_conf[Datasets][TagName][DatasetName]
        return self.Tag_Dataset

    def get_External_Datasets(self):
        self.Ext_Data = self.yaml_conf["External_Datasets"]
        return self.Ext_Data

    def get_External_Tags(self):
        self.Ext_Tags = self.yaml_conf["External_Datasets"]['External_Tags']
        return self.Ext_Tags

    def get_Feature_Threshold(self):
        self.fThreshold = self.yaml_conf['Feature_Selection_Threshold']
        return self.fThreshold


class Import_Dataset_Config:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.import_conf = self.yaml_conf['Import_Data']

    def get_import_conf(self):
        return self.import_conf

    def get_inputFiles(self):
        self.inputfiles = self.import_conf['Input']
        return self.inputfiles

    def get_Tags(self):
        self.tags = self.import_conf['Tags']
        return self.tags


class Database_Config:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.database_conf = self.yaml_conf['Database_Config']

    def get_database_conf(self):
        return self.database_conf

    def get_p_username(self):
        self.p_username = self.database_conf["p_username"]
        return self.p_username

    def get_p_password(self):
        self.p_password = self.database_conf["p_password"]
        return self.p_password

    def get_p_host(self):
        self.p_host = self.database_conf["p_host"]
        return self.p_host

    def get_p_service(self):
        self.p_service = self.database_conf["p_service"]
        return self.p_service

    def get_p_port(self):
        self.p_port = self.database_conf["p_port"]
        return self.p_port


class Dataset_Configs:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()

    def get_Target_Column(self):
        self.Target_Column = self.yaml_conf["Target_Column"]
        return self.Target_Column

    def get_Tag_Column(self):
        self.Tag_Column = self.yaml_conf["Tag_Column"]
        return self.Tag_Column

    def get_Feature_Threshold(self):
        self.fThreshold = self.yaml_conf['Feature_Selection_Threshold']
        return self.fThreshold

    def get_Upload_Tags(self, DatasetName):
        self.Upload_Tags = self.yaml_conf[DatasetName]["Upload_Tags"]
        return self.Upload_Tags

    def get_External_Datasets(self):
        self.Ext_Data = self.yaml_conf["External_Datasets"]
        return self.Ext_Data

    def get_Datasets(self):
        self.Ext_Data = self.yaml_conf["Datasets"]
        return self.Ext_Data

    def get_Tag(self, Datasets, TagName):
        self.Tag = self.yaml_conf[Datasets][TagName]
        return self.Tag

    def get_Tag_Datasets(self, Datasets, TagName, DatasetName):
        self.Tag_Dataset = self.yaml_conf[Datasets][TagName][DatasetName]
        return self.Tag_Dataset


class Create_Labels_Configs:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.label_conf = self.yaml_conf['Create_Labels']
        self.dataset_conf = self.label_conf['Upload_Tags']

    def get_Upload_Tags(self):
        return self.dataset_conf

    def get_inputFiles(self, dataset):
        dataset_conf = self.label_conf[dataset]
        return dataset_conf['Input']

    def get_Tags(self, dataset):
        dataset_conf = self.label_conf[dataset]
        return dataset_conf['Tags']

    def get_Categories(self, dataset):
        dataset_conf = self.label_conf[dataset]
        return dataset_conf['Category']

    def get_new_Column_Categories(self, dataset):
        dataset_conf = self.label_conf[dataset]
        return dataset_conf['New_Category_Column']

    def get_refColumn(self, dataset):
        dataset_conf = self.label_conf[dataset]
        return dataset_conf['Refcolumn']

    def get_output(self, dataset):
        dataset_conf = self.label_conf[dataset]
        return dataset_conf['Output']


class CombineDataset_Configs:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.combine_conf = self.yaml_conf['Combine_Datasets']
        self.dataset_conf = self.combine_conf['Upload_Data']

    def get_Upload_Datasets(self):
        return self.dataset_conf

    def get_combineDataset_step_conf(self):
        return self.combine_conf

    def get_inputFiles(self, dataset):
        dataset_conf = self.combine_conf[dataset]
        return dataset_conf['Input']

    def get_combine_Tags(self, dataset):
        dataset_conf = self.combine_conf[dataset]
        return dataset_conf['Combine_Tags']

    def get_Event_Threshold(self, dataset):
        dataset_conf = self.combine_conf[dataset]
        return dataset_conf['Event_Threshold']

    def get_Output(self, dataset):
        dataset_conf = self.combine_conf[dataset]
        return dataset_conf['Output']


class Split_Train_Test_Configs:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.split_conf = self.yaml_conf['Split_Train_test']

    def get_Split_step_conf(self):
        return self.split_conf

    def get_inputFiles(self):
        return self.split_conf['Input']

    def get_drop_Features(self):
        return self.split_conf['Drop_Features']

    def get_referance_Column(self):
        return self.split_conf['Referance_column']

    def get_selected_data(self):
        return self.split_conf['Selected_data']

    def get_type(self):
        return self.split_conf['Type']

    def get_test_size(self):
        return self.split_conf['Test_size']

    def get_stratify(self):
        return self.split_conf['Stratify']

class Feature_Selection_Configs:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.feature_selection_conf = self.yaml_conf['Feature_Selection']
        self.feature_selection_method_conf = self.feature_selection_conf['Feature_Selection_Method']
    
    def get_FeatureSelection_conf(self):
        return self.feature_selection_conf

    def get_FeatureSelection_method_conf(self):
        return self.feature_selection_method_conf

    def get_inputFiles(self):
        return self.feature_selection_conf['Input']

    def get_metaFile(self):
        return self.feature_selection_conf['Metadata']

    def get_Feature_Selection_Method_Config(self, TagName):
        self.TagConf = self.feature_selection_method_conf[TagName]
        return self.TagConf

    def get_Feature_Selection_Threshold(self):
        return self.feature_selection_conf['Feature_Selection_Threshold']

    def get_Feature_Selection_Method(self):
        return self.feature_selection_conf['Feature_Selection_Method']

    def get_Classification(self):
        return self.feature_selection_conf['Classification']

    def get_Average(self):
        return self.feature_selection_conf['Average']

class Feature_Extraction_Configs:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.feature_extraction_conf = self.yaml_conf['Feature_Extraction']
        self.feature_extraction_method_conf = self.feature_extraction_conf['Feature_Extraction_Method']

    def get_Feature_Extraction_step_conf(self):
        return self.feature_extraction_conf

    def get_Feature_Extraction_method_conf(self):
        return self.feature_extraction_method_conf

    def get_Feature_Extraction_List(self):
        return self.feature_extraction_conf['Feature_Ext_List']

    def get_Feature_Extraction_Method_Config(self, TagName):
        self.TagConf = self.feature_extraction_method_conf[TagName]
        return self.TagConf

    def get_inputFiles(self):
        return self.feature_extraction_conf['Input']

    def get_Method_inputFiles(self,TagName):
        conf=self.get_Feature_Extraction_Method_Config(TagName)
        return conf['Input']

    def get_metaFile(self):
        return self.feature_extraction_conf['Metadata']

    def get_Feature_Extraction_Threshold(self):
        return self.feature_extraction_conf['Feature_Extraction_Threshold']

    def get_Feature_Extraction_Method(self):
        return self.feature_extraction_conf['Feature_Extraction_Method']

    def get_Normalize_Feature_List(self):        
        conf=self.get_Feature_Extraction_Method_Config('Normalize')
        return conf['normalize_feature_list']

    def get_Average(self):
        return self.feature_extraction_conf['Average']

class Imputation_Configs:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.imputation_conf = self.yaml_conf['Imputation']
        self.imputation_list_conf = self.imputation_conf['feature_fill']
    
    def get_imputation_conf(self):
        return self.imputation_conf

    def get_inputFiles(self):
        return self.imputation_conf['Input']

    def get_fill_list(self):
        return self.imputation_conf['feature_fill']

    def get_list_conf(self,Tag):
        return self.imputation_conf[Tag]

class Model_Configs:
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.model_conf = self.yaml_conf['Model_Parameters']

    def get_inputFiles(self):
        return self.model_conf['Input']

    def get_models_path(self):
        parent_path = self.yaml_conf['Parent_path']
        model_path = os.path.join(parent_path, 'data', '06_models')
        # self.models_path=self.yaml_conf['Parent_path']+"data/06_models/"
        return model_path

    def get_models_params(self, type):
        self.model_params = self.yaml_conf["Model_Parameters"][type]
        return self.model_params

    def get_grid_search(self, type):
        grid_search_ = self.yaml_conf["Model_Parameters"][type]["grid_search"]
        return grid_search_

    def get_number_of_cv_folds(self, type):
        # print(self.yaml_conf["Model_Parameters"][type])
        cv = self.yaml_conf["Model_Parameters"][type]["cv"]
        if cv != None and isinstance(cv, int):
            self.number_of_cv_folds = cv
        else:
            self.number_of_cv_folds = 5
        return self.number_of_cv_folds

    def get_scoring_type(self, type):
        scoring = self.yaml_conf["Model_Parameters"][type]["scoring"]
        return scoring


class Upload_Download_Pickle():
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.parent_path = self.yaml_conf['Parent_path']

    @staticmethod
    def download_pickle(path, filename):
        path = os.path.join(path, filename)        
        if os.path.exists(path):
            with open(path, 'rb') as config_file:
                pic = pickle.load(config_file)
        else:
            pic="no_file_exist"
        return pic

    def download_pickle_files(self, path, filenames):
        pickle_list = []
        for i in filenames:
            s = self.download_pickle(path, i)
            pickle_list.append(s)
        return tuple(pickle_list)

    @staticmethod
    def save_dataset_pickle(path, filename, output):
        path = os.path.join(path, filename)
        with open(path, 'wb') as config_file:
            pickle.dump(output, config_file)


class Logging_Config():
    def __init__(self):
        self.yaml_parser = YamlParser()
        self.yaml_conf = self.yaml_parser.get_config_paths()
        self.logging_conf = self.yaml_conf['Logging_Config']

    def get_logger(self, logger):
        return self.logging_conf[logger]

    def get_filename(self, logger):
        return self.logging_conf['logger'][logger]['filename']

    def get_format(self, logger):
        return self.logging_conf['logger'][logger]['format']

# Function to load yaml configuration file


def load_config(CONFIG_PATH, config_name):
    """
    Load a YAML configuration file.
    Parameters
    ----------
    config_name : open(os.path.join(CONFIG_PATH,  config_name), 'r', encoding='utf8')

    Returns
    -------
    cfg : dict
    """
    with open(os.path.join(CONFIG_PATH, config_name), 'r', encoding='utf8') as file:
        config = yaml.safe_load(file)

    return config


def concat_data_set(datalist):
    # for data in datalist:
    concatdata = pd.concat(datalist, ignore_index=True)
    return concatdata

# Convert dataframe to list according to reference column unique values
# türbin datalarını birleştirmek üzere kullanılır.


def convert_dataframe_to_list(df, refcolumn):
    X_list = []
    for i in df[refcolumn].unique():
        ref_df = df[df[refcolumn] == i].copy()
        ref_df.drop(refcolumn, axis='columns')
        X_list.append(ref_df)
    return X_list
