import os
import sys
import pandas as pd
import logging
from src.utils.utils_p import YamlParser, Config_Paths, Dataset_Configs, Upload_Download_Pickle
sys.path.append('..')


""""""
# pip install xlrd
# pip install pandas
# pip install numpy


class CollectDataFromCSV():
    """CollectDataFromCSV"""

    def __init__(self):
        """init"""
        self.yaml_file = YamlParser()
        self.config = self.yaml_file.get_yaml_file()

        # Input & Output Paths:
        self.config_paths = Config_Paths()
        self.raw_path = self.config_paths.get_raw_path()
        self.parent_path = self.config_paths.get_parent_path()

        # Dataset_Configs
        self.dataset_config = Dataset_Configs()
        self.dataset_tags = Dataset_Configs().get_Upload_Tags('Datasets')
        self.external_tags = Dataset_Configs().get_Upload_Tags('External_Datasets')
        logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('CollectDataFromCSV')  # __name__=projectA.moduleB


    def upload_all_datasets(self):
        """upload datasets: upload Datasets according to tags defined importcsv.yaml file
            Örnek Configler:
            Datasets:
                Upload_Tags: ['T02','T03']
                T02:... -- dataset config bilgilerini içerir
            External_Datasets:
                Upload_Tags: ['Component']
                Component:... --dataset config bilgilerini içerir
        """
        # print('upload these datasets',self.config["Upload_list_all"]["Datasets"]["Upload_Tags"])
        self.logger.warning('upload these datasets %s' % self.dataset_tags)
        #print('upload these datasets', self.dataset_tags)
        for tag_name in self.dataset_tags:
            dataset_conf = Dataset_Configs().get_Tag('Datasets', tag_name)
            self.upload_datasets_tagname(dataset_conf, tag_name)

        # print('upload these datasets',self.config["Upload_list_all"]["External_Datasets"]["Upload_Tags"])
        self.logger.warning('upload external datasets %s' % self.external_tags)
        #print('upload these datasets', self.external_tags)
        for tag_name in self.external_tags:
            ext_dataset_conf = Dataset_Configs().get_Tag('External_Datasets', tag_name)
            self.upload_datasets_tagname(ext_dataset_conf)

    def upload_external_datasets(self):
         # print('upload these datasets',self.config["Upload_list_all"]["External_Datasets"]["Upload_Tags"])
        self.logger.warning('upload external datasets %s' % self.external_tags)
        #print('upload these datasets', self.external_tags)
        for tag_name in self.external_tags:
            ext_dataset_conf = Dataset_Configs().get_Tag('External_Datasets', tag_name)
            self.upload_datasets_tagname(ext_dataset_conf) 
                  
    def upload_datasets_tagname(self, config, tag_name=''):
        """upload dataset: upload Datasets according to tags defined importcsv.yaml file
            -- Datasetlerden hangilerinin upload edileceğine Upload_Tags'e belirlenen datalara göre 
            eğer T02 ya da T03 yuklenecekse bu verilerin bilgilerinin eklenmesi gerekir. 
            Örnek Config:
            Datasets:
                Upload_Tags: ['T02','T03']
                T02:
                    Combine:
                        combine_datasets: ['Logs_normal_fill','Faults']
                    Dataset_Tags: ['Dataset1','Dataset2']
                    Dataset1:...
                    Dataset2:...  
        """
        datalist = config['Dataset_Tags']
        for data in datalist:
            # load dataset
            dataset = self.load_dataset(config=config, dataset_name=data)
            # save as a pickle file
            name = config[data]['Name']
            Upload_Download_Pickle().save_dataset_pickle(
                self.raw_path, name + tag_name, dataset)

    def load_dataset(self, config, dataset_name):
        """Tagde verilen veri setini yükleme kısmı:
            --  Bu işlem için Dataset_Tags bilgisine ihtiyaç var. Örnekte verilen T02 veri setini 
            oluşturmak için birden fazla veri setine ihtiyaç duyduğumuz bir durumlar olabiliyor. 
            Özellikle feature set ve label set ayrı dosyalarda ise bu durum oluşabiliyor. 
            Bu nedenle Dataset_Tags ekledik. Aşağıdaki örneğe göre Dataset1, Dataset2 bilgilerini
            upload edilecek demek oluyor. 
            -- Dataset1 ve 2nin upload edilebilmesi için veri set bilgilerinin eklenmesi gerekir. 
            Bu bilgiler sırasıyla Dosyanın ismi Name: Logs, yüklenecek her bir sheet için Path_list, 
            File_List, Sheet_list ve Range_list'e ihtiyacımız var. refcolumn olarak 'Unnamed: 0' verilmiş
            örnekte bu bizim index kolonumuz. Bu kolon verisetimizde aslında Period kolonu. 
            Refcolumn'un aynı zamanda amacı Dataset1 ve Dataset2 veri setlerini birleştirmek için Dataset1 
            tarafında kullanılacak kolon olması. O nedenle bu kolonu ona göre vermek dogru olur.
            Örnek Config: 
            Datasets:
                Upload_Tags: ['T02','T03']
                T02:
                    Dataset_Tags: ['Dataset1','Dataset2']
                    Dataset1:
                        -- Dataset ismi pickle file oluştururken kullanılacak
                        Name: 'Logs' 
                        --data klasörünün altındaki path bilgisi
                        Path_list: ['log','log','log','log'] 
                        -- data klasörünün altındaki excel dosyası
                        File_list: ['T02.xls','T02.xls','T02.xls','T02.xls'] 
                         -- sheet ismi
                        Sheet_list: ['Gen.Operation Data','Gearbox Data','Main Bearing Data','Pitch System Data']
                        -- hangi excel satırından itibaren veri alınacak bilgisi
                        Range_list: [[0,11],[0,11],[0,11],[0,11]] 
                        -- index/referans kolon
                        refcolumn: 'Unnamed: 0' 
                        -- data preprocessingde kullanılacak featurelar 
                        feature_fill:... 
                    Dataset2:...      
        """
        # Get Dataset config from yaml file:
        dataset = config[dataset_name]
        # check if dataset config is not null
        if dataset != None:
            if (dataset['Path_list'] is not None and dataset['File_list'] != None and dataset['Sheet_list'] != None and
                    dataset['Range_list'] != None):
                if len(dataset['Path_list']) > 1:
                    if dataset['refcolumn'] is not None:
                        # If there are more than one sheet with same columns use upload_multiple_sheets
                        data = self.upload_multiple_sheets(
                            self.parent_path, dataset['Path_list'], dataset['File_list'], dataset['Sheet_list'],
                            dataset['Range_list'], dataset['refcolumn'],dataset['Type'])
                    else:
                        raise ValueError(
                            "No refcolumn config exists on yamlfile")
                elif len(dataset['Path_list']) == 1:
                    # If there is only one sheet to upload use upload_single_sheet
                    data = self.upload_single_sheet(
                        self.parent_path, dataset['Path_list'][0], dataset['File_list'][0], dataset['Sheet_list'][0],
                        skiprows=dataset['Range_list'])
                else:
                    # print('No Path exist')
                    raise ValueError("No Path_List Config exist")
            else:
                print('Please Check All Dataset Configs Path_list:',
                      dataset['Path_list'], 'File_list: ', dataset['File_list'], 'Sheet_list:', dataset['Sheet_list'],
                      'Range_list:', dataset['Range_list'], 'refcolumn:', dataset['refcolumn'])
                raise ValueError('One of Dataset config is missing!')
        else:
            raise ValueError('Please Define Dataset')
        data = self.drop_columns(data, config, dataset_name)
        return data

    def upload_multiple_sheets(self, parent_path, path_list, file_list, sheet_list, range_list, refcolumn,type):
        """ birden fazla excel sheet verisini alıp birleştirip çift satırları atıp birleştirme işlemi
        """
        # create List with sheets
        combined_dataf = []
        for i, item in enumerate(path_list):
            # for i in range(len(path_list)):
            # upload single sheet
            dataf = self.upload_single_sheet(
                parent_path=parent_path, path=item, filename=file_list[i], sheetname=sheet_list[i], skiprows=range_list[i])
            # print('dataf.columns',dataf.columns)
            # remove duplicate columns according to refcolumn
            dataf = self.remove_key_columns_duplicate_rows(dataf, refcolumn)
            # append uploadded dataframe/sheet to List
            combined_dataf.append(dataf)

        # pre define a dataframe as 'dataf'
        dataf = combined_dataf[0].set_index(refcolumn)
        print('file_list : ',file_list[0],' sheet_name : ',sheet_list[0], ' shape:',dataf.shape)
        if type=='merge':
            for i in range(1, len(combined_dataf)):
                # merge all list according to reference column
                dataf = dataf.merge(combined_dataf[i].set_index(
                    refcolumn), on=refcolumn, how='left')
        elif type=='concat':
            for i in range(1, len(combined_dataf)):
                # merge all list according to reference column
                print('file_list : ',file_list[i],' sheet_name : ',sheet_list[i], ' shape:',dataf.shape)
                print('combined_dataf.shape: ',combined_dataf[i].shape)
                dataf=pd.concat([dataf,combined_dataf[i].set_index(refcolumn)])
                #dataf = dataf.concat()
        dataf.isna().sum()
        return dataf

    def upload_single_sheet(self, parent_path, path, filename, sheetname, skiprows=None):
        """sheetdeki veriyi yukleme işlemi
        -- read excel
        -- çift satırları ayıklar
        -- full nan olan kolonları ayıklar
        -- na satırları da ayıklar
        """
        #print('parent_path : ', parent_path, 'path : ', path, 'filename : ', filename, 'sheetname : ', sheetname,
        #      'skiprows : ', skiprows)
        self.logger.warning('upload_single_sheet : ')
        self.logger.warning('parent_path : %s, path : %s filename : %s sheetname : %s skiprows : %s', parent_path,  path, filename, sheetname,skiprows)
        addres = os.path.join(parent_path, 'data', '01_raw', path, filename)
        # read single sheet
        dataf = pd.read_excel(addres, sheet_name=sheetname,
                              skiprows=range(skiprows[0], skiprows[1]))

        # -- çift satırları ayıklar
        # remove duplicate rows
        dataf = self.remove_all_columns_duplicate_rows(dataf)
        #self.logger.warning('remove_all_columns_duplicate_rows - {}'.format(dataf.head().to_string()))
        # -- full nan olan kolonları ayıklar
        # remove nan columns
        self.logger.warning('drop_full_nan_columns: ')
        dataf = self.drop_full_nan_columns(dataf)
        #self.logger.warning('after drop : %s ' % dataf.isnull().sum())
        # -- na satırları da ayıklar
        # remove nan rows
        dataf = dataf.dropna(how='all')

        # print(dataf.columns)
        return dataf

    @staticmethod
    def remove_key_columns_duplicate_rows(dataf, refcolumn):
        """remove_key_columns_duplicate_rows"""
        dataf = dataf[~dataf[refcolumn].duplicated()]
        return dataf

    @staticmethod
    def drop_full_nan_columns(dataf):
        """drop_full_nan_columns"""
        for i in dataf.columns:
            if dataf[i].isnull().all() == True:
            #if dataf[i].isnull().all() is True:
                dataf = dataf.drop(i, axis='columns')
        return dataf

    @staticmethod
    def remove_all_columns_duplicate_rows(dataf):
        """remove_all_columns_duplicate_rows"""
        dataf = dataf[~dataf.duplicated()]
        return dataf

    @staticmethod
    def upload_config_paths(dataf):
        """upload_config_paths"""
        dataf = dataf[~dataf.duplicated()]
        return dataf

    @staticmethod
    def get_drop(config, dataset):
        """get_drop"""
        for i in config[dataset]:
            drop = False
            if i == 'drop':
                if config[dataset][i]['value']:
                    drop = True
        return drop

    def drop_columns(self, data, config, dataset_name):
        """drop_columns"""
        if self.get_drop(config, dataset_name):
            try:
                data = data.drop(config[dataset_name]
                                 ['drop']['refcolumns'], axis='columns')
                #self.logger.warning('data columns after drop : %s' % data.columns)
            except:
                print(config[dataset_name]['drop'])
                print(data.columns)
                raise ValueError(
                    'refcolumns are not exist! Please check config')
        return data
