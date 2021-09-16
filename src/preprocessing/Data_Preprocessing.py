import sys
from src.utils.utils_p import YamlParser, Dataset_Configs
from src.preprocessing.Clean_Rename import Clean_Rename
from src.preprocessing.Create_Labels import Create_Labels
sys.path.append('../..')


class Data_Preprocessing():
    def __init__(self):
        self.yaml_file = YamlParser()
        self.config = self.yaml_file.get_yaml_file()
        self.dataset_config = Dataset_Configs()

    def preprocessing(self):
        """Clean & Normalize Logs(Features) data & Combine Logs and Faults for each tags"""
        # config=self.config
        dataset_list = self.dataset_config.get_Upload_Tags('Datasets')
        # for TagName in config["Upload_list_all"]["Datasets"]["Upload_Tags"]:
        for tagname in dataset_list:
            dataset_conf = self.dataset_config.get_Tag_Datasets(
                'Datasets', tagname, 'Dataset1')
            # self.Single_Preprocessing(dataset_conf,TagName)
            Clean_Rename().single_clean_rename(dataset_conf, tagname)
            # Combine Logs & Faults
            # combine_conf=self.dataset_config.get_Combine_Datasets(TagName)
            Clean_Rename().combine(dataset_conf, tagname)
            # Event referans alarak once binary sonra multiclass sonra da onehot labellar oluştur.
            #Create_Labels().single_create_labels_with_category('Logs_Faults',tagname, 'Events','FaultComponent')
        Create_Labels().create_labels_with_category()
        

            
    @staticmethod
    def single_preprocessing(dataset_conf, tagname):
        """Clean & Normalize Logs(Features) data & Combine Log and Fault for single dataset_conf"""
        #####Clean & Normalize Logs(Features) data & Combine Logs and Faults#####
        #Clean and Normalize
        Clean_Rename().single_clean_rename(dataset_conf, tagname)
        # Combine Logs & Faults
        Clean_Rename().combine(dataset_conf, tagname)
        # Event referans alarak once binary sonra multiclass sonra da onehot labellar oluştur.
        Create_Labels().single_create_labels_with_category('Logs_Faults',tagname, 'Events','FaultComponent')
