import sys
from src.data.CollectDataFromCSV import CollectDataFromCSV
from src.utils.utils_p import Dataset_Configs
sys.path.append('..')


class Upload():
    """Upload Datasets according to Each Tag"""
    # print('upload these datasets',self.config["Upload_list_all"]["Datasets"]["Upload_Tags"])
    csv = CollectDataFromCSV()
    print('upload these datasets', csv.dataset_tags)
    for TagName in csv.dataset_tags:
        dataset_conf = Dataset_Configs().get_Tag('Datasets', TagName)
        #csv.upload_datasets(dataset_conf, TagName)
        csv.upload_datasets_tagname(dataset_conf, TagName)

    # print('upload these datasets',self.config["Upload_list_all"]["External_Datasets"]["Upload_Tags"])
    print('upload these datasets', csv.external_tags)
    for TagName in csv.external_tags:
        ext_dataset_conf = Dataset_Configs().get_Tag('External_Datasets', TagName)
        csv.upload_datasets_tagname(ext_dataset_conf)
