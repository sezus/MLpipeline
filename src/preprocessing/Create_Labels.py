import pandas as pd
import logging
from sklearn import preprocessing
from src.utils.utils_p import YamlParser, Config_Paths, Upload_Download_Pickle,Create_Labels_Configs,Logging_Config
# pip install sklearn


class Create_Labels():
    def __init__(self):
        self.yaml_file = YamlParser()
        self.config = self.yaml_file.get_yaml_file()
        self.config_paths = Config_Paths()
        self.raw_path = self.config_paths.get_raw_path()
        self.intermediate_path = self.config_paths.get_intermediate_path()

        self.label_conf= Create_Labels_Configs()
        self.datasets=self.label_conf.get_Upload_Tags()

        self.log_cfg=Logging_Config()        
        logging.basicConfig(filename=self.log_cfg.get_filename('sampleLogger'), filemode='w', format=self.log_cfg.get_format('sampleLogger'))
        self.logger = logging.getLogger('Create_Labels')  # __name__=projectA.moduleB    

    def create_labels_with_category(self):
         dataset_list = self.datasets    
         for data in dataset_list:
            tags=self.label_conf.get_Tags(data)
            for tag in tags:
                print('tag',tag)
                self.single_create_labels(data=data,tagname=tag)



    def single_create_labels_with_category(self, filename, tagname, eventscolumn,category):
        # raw_path=self.raw_path
        # intermediate_path=self.intermediate_path

        # Download Pickle Files
        logs_faults = Upload_Download_Pickle().download_pickle(
            self.intermediate_path, filename + tagname)

        #####Create Labels#####
        # Event referans alarak once binary sonra multiclass sonra da onehot labellar oluştur.
        # Yeni Labellar oluşturmadan önceki kolon sayısı before_size_df
        default_size = logs_faults.shape[1]

        # Target Column Faults
        faults = logs_faults[logs_faults.columns[range(
            default_size-1, default_size)]]

        # 1.Eventler için Binary Multiclass ve Onehot labellar oluştur.
        logs_faults = self.create_binary_multiclass_onehot_labels(
            logs_faults, eventscolumn)

        # 2 Mevcut Eventleri yeniden belli  Kategorilere göre sınıflandırma isteyebiliriz.
        # Bunun için Logs ve Mevcut Faulttan farklı bir excel ekleyip bu yeni gelen Kategoriye göre
        # Yeni Labellar oluşturmak isteyebiliriz.
        fault_component = Upload_Download_Pickle().download_pickle(
            self.raw_path, category)
        FC = pd.DataFrame(fault_component)
        if FC.empty != True:
            # FaultComponent=FaultComponent.rename(columns=['Number', 'Braking', 'Main Category', 'Events'])
            fault_component = self.create_category_column_from_events(
                faults, fault_component, 'COMPONENT', eventscolumn)
            # Yeni gelen Categoryleri Oluşan Eventlerle birlestirip
            # Eventlerin karşılık geldiği Componentlere ait binary & multiclass & onehot labellar oluşturma işlemi:
            self.logger.debug('Logs not shape 0')
            logs_faults = self.create_component_labels(
                logs_faults, eventscolumn, fault_component, eventscolumn, 'COMPONENT')

        # Son olarak Oluşturulan Labelları farklı "Label"
        # DataFrame'inde toplayıp "Features" ve "Labels" olarak ayırırızı
        # Labels ayrı bir Dataframe olarak tut: "Labels"

        labels = logs_faults[logs_faults.columns[range(
            (default_size-1), logs_faults.shape[1])]]
        labels['Period'] = logs_faults['Period']
        # Featureları ayrı bir Dataframe olarak tut: "Features"
        features = logs_faults.drop(logs_faults.columns[range(
            default_size, logs_faults.shape[1])], axis='columns')
        # Save Features and Labels:
        Upload_Download_Pickle().save_dataset_pickle(
            self.intermediate_path, 'Labels' + tagname, labels)
        Upload_Download_Pickle().save_dataset_pickle(
            self.intermediate_path, 'Features' + tagname, features)

    def single_create_labels(self, data,tagname):
        # raw_path=self.raw_path
        # intermediate_path=self.intermediate_path
        filename=self.label_conf.get_inputFiles(data)
        eventscolumn=self.label_conf.get_refColumn(data)
        category=self.label_conf.get_Categories(data)
        new_column=self.label_conf.get_new_Column_Categories(data)
        output=self.label_conf.get_output(data)
        
        print('filename',filename[0])
        # Download Pickle Files
        logs_faults = Upload_Download_Pickle().download_pickle(
            self.intermediate_path, filename + tagname )


        #####Create Labels#####
        # Event referans alarak once binary sonra multiclass sonra da onehot labellar oluştur.
        # Yeni Labellar oluşturmadan önceki kolon sayısı before_size_df
        default_size = logs_faults.shape[1]

        # Target Column Faults
        faults = logs_faults[logs_faults.columns[range(
            default_size-1, default_size)]]

        # 1.Eventler için Binary Multiclass ve Onehot labellar oluştur.
        logs_faults = self.create_binary_multiclass_onehot_labels(
            logs_faults, eventscolumn)

        # 2 Mevcut Eventleri yeniden belli  Kategorilere göre sınıflandırma isteyebiliriz.
        # Bunun için Logs ve Mevcut Faulttan farklı bir excel ekleyip bu yeni gelen Kategoriye göre
        # Yeni Labellar oluşturmak isteyebiliriz.
        logs_faults=self.create_labels_for_new_category(category,faults,logs_faults,eventscolumn,new_column)
        print('logs_faults', logs_faults.shape)
        # Son olarak Oluşturulan Labelları farklı "Label"
        # DataFrame'inde toplayıp "Features" ve "Labels" olarak ayırırızı
        # Labels ayrı bir Dataframe olarak tut: "Labels"
        labels,features=self.divide_Labels_Features(default_size=default_size,logs_faults=logs_faults)
        #self.divide_save_Labels_Features(self,tagname,default_size,logs_faults)
        # Save Features and Labels:
        self.save_Features_Labels(tagname,labels,features,output)

    def create_labels_for_new_category(self,category,faults,logs_faults,eventscolumn,new_column):
        fault_component = Upload_Download_Pickle().download_pickle(
            self.raw_path, category)

        FC = pd.DataFrame(fault_component)
        if FC.empty != True:
            # FaultComponent=FaultComponent.rename(columns=['Number', 'Braking', 'Main Category', 'Events'])
            fault_component = self.create_category_column_from_events(
                faults, fault_component, new_column, eventscolumn)
            # Yeni gelen Categoryleri Oluşan Eventlerle birlestirip
            # Eventlerin karşılık geldiği Componentlere ait binary & multiclass & onehot labellar oluşturma işlemi:
            self.logger.debug('Logs not shape 0')
            logs_faults = self.create_component_labels(
                logs_faults, eventscolumn, fault_component, eventscolumn, new_column)
        return logs_faults

    def divide_Labels_Features(self,default_size,logs_faults):
        # Son olarak Oluşturulan Labelları farklı "Label"
        # DataFrame'inde toplayıp "Features" ve "Labels" olarak ayırırızı
        # Labels ayrı bir Dataframe olarak tut: "Labels"

        labels = logs_faults[logs_faults.columns[range(
            (default_size-1), logs_faults.shape[1])]]
        print('label',labels.columns)
        print('label',labels.shape)
        labels['Period'] = logs_faults['Period']
        # Featureları ayrı bir Dataframe olarak tut: "Features"
        features = logs_faults.drop(logs_faults.columns[range(
            default_size, logs_faults.shape[1])], axis='columns')

        return labels,features

    def save_Features_Labels(self, tagname,labels,features,output):
        self.logger.debug('Save Features')
        Upload_Download_Pickle().save_dataset_pickle(
            self.intermediate_path, output[0] + tagname, labels)
        Upload_Download_Pickle().save_dataset_pickle(
            self.intermediate_path, output[1]  + tagname, features)

    def create_labels(self, logs_faults, eventscolumn, fault_component=None):
        #####Create Labels#####
        # Event referans alarak once binary sonra multiclass sonra da onehot labellar oluştur.
        # Yeni Labellar oluşturmadan önceki kolon sayısı before_size_df
        default_size = logs_faults.shape[1]
        # Target Column Faults
        print('default_size:',default_size)
        faults = logs_faults[logs_faults.columns[range(default_size-1, default_size)]]
        # 1.Eventler için Binary Multiclass ve Onehot labellar oluştur.
        logs_faults = self.create_binary_multiclass_onehot_labels(
            logs_faults, eventscolumn)

        # 2 Mevcut Eventleri yeniden belli  Kategorilere göre sınıflandırma isteyebiliriz.
        # Bunun için Logs ve Mevcut Faulttan farklı bir excel ekleyip bu yeni gelen Kategoriye göre
        # Yeni Labellar oluşturmak isteyebiliriz.
        FC = pd.DataFrame(fault_component)
        if FC.empty != True:
            self.logger.debug('FC True %s' %fault_component)
            #FaultComponent=FaultComponent.rename(columns=['Number', 'Braking', 'Main Category', 'Events'])
            fault_component = self.create_category_column_from_events(
                faults, FC, 'COMPONENT', eventscolumn)
            self.logger.debug('FaultComponent.columns %s' %fault_component.columns)
            
            # Yeni gelen Categoryleri Oluşan Eventlerle birlestirip
            # Eventlerin karşılık geldiği Componentlere ait binary & multiclass & onehot labellar oluşturma işlemi:
            logs_faults = self.create_component_labels(
                logs_faults, eventscolumn, fault_component, 'Events', 'COMPONENT')

        # Son olarak Oluşturulan Labelları farklı "Label"
        # DataFrame'inde toplayıp "Features" ve "Labels" olarak ayırırızı
        # Labels ayrı bir Dataframe olarak tut: "Labels"
        #print('Logs_Faults.shape[1]', logs_faults.shape[1])
        self.logger.debug('logs_faults.shape[1] %s' %logs_faults.shape[1])
        Labels = logs_faults[logs_faults.columns[range(
            default_size, logs_faults.shape[1])]]
        Labels['Period'] = logs_faults['Period']
        # Featureları ayrı bir Dataframe olarak tut: "Features"
        Features = logs_faults.drop(logs_faults.columns[range(
            default_size, logs_faults.shape[1])], axis='columns')
        return Features, Labels

    @staticmethod
    def create_binary(df, refcolumn):
        binary = refcolumn+'_binary'
        df[binary] = df[refcolumn].copy()
        df.loc[df[refcolumn] != 0, binary] = 1
        return df

    @staticmethod
    def create_multiclass( df, refcolumn):
        multiclass = refcolumn+'_multiclass'
        le = preprocessing.LabelEncoder()
        df[refcolumn] = df[refcolumn].astype(str).copy()
        le.fit(df[refcolumn])
        df[multiclass] = le.transform(df[refcolumn])
        return df
    @staticmethod
    def create_onehot(df, refcolumn):
        ref = df[refcolumn]
        df = pd.get_dummies(df, columns=[refcolumn])
        df[refcolumn] = ref
        return df

    def create_binary_multiclass_onehot_labels(self,df, refcolumn):
        df = self.create_binary(df, refcolumn)
        df = self.create_multiclass(df, refcolumn)
        df = self.create_onehot(df, refcolumn)
        return df

    def create_component_labels(self, df, refcolumn, categorydf, categoryrefcolumn, newcategory):
        """
        df:Dataframe with refcolumn
        refcolumn:Name of Target Column will reclassify with new Category
        categorydf:Dataframe has values both refcolumn and new category values
        categoryrefcolumn:Name of Target Column on faultdf
        newcategory: Name of new Category Column Name
        """
        # Grep unique category names
        component = categorydf[newcategory].unique()
        # create a new column for new category referances as 'Components'
        df['Components'] = df[refcolumn]
        # For each unique component
        for i in component:
            # Collect Unique Events belong the new component
            ComponentEvents = categorydf[categorydf[newcategory]
                                         == i][categoryrefcolumn].unique()

            # 1. Create Binary Component Label Column
            # 1.1 Create new column from each unique component assing '0'
            df[i] = 0
            # 1.2 Assign '1' only the new component Event ones
            df.loc[df[refcolumn].isin(ComponentEvents), i] = 1

            # 2.Create Component column multiclass
            Even = str(i)+'Ev'
            self.logger.debug('Event %s' %Even)
            df[Even] = 0
            df.loc[df[refcolumn].isin(ComponentEvents), Even] = df.loc[df[refcolumn].isin(
                ComponentEvents)].Events
            self.logger.debug('Create Component column multiclass: %s' %df.columns)

            # 3.Create Multiclass new component Event
            self.create_multiclass(df, Even)
            self.logger.debug('3.Create Multiclass new component Event: %s' %df.columns)
            df.loc[df[refcolumn].isin(ComponentEvents), 'Components'] = i
        self.create_multiclass(df, 'Components')
        return df

    # Sınıflandırılma tahmininde kullanılmak üzere Label target kolonunundan, Target kolonundaki Eventlerin
    # ait oldugu Kategorilere göre yeni bir target kolonu oluşturmak için kullanılır.
    # This method create a new target column from new categorycolumn by matching Event column
    def create_category_column_from_events(self, dfEvent, dfCategory, categorycolumn, eventcolumn):
        self.logger.debug('dfEvent.columns: %s' %dfEvent.columns)
        self.logger.debug('dfCategory.columns: %s' %dfCategory.columns)
        self.logger.debug('dfCategory[[eventcolumn].unique(): %s' %dfCategory[eventcolumn].unique())
        self.logger.debug('dfEvent[eventcolumn].unique(): %s' % dfEvent[eventcolumn].unique())

        # Fault(Target) değişken ile Event listesini merge et
        dfCategoryEvents = pd.merge(
            dfEvent, dfCategory[[categorycolumn, eventcolumn]], how='left',  on=eventcolumn)
        #self.logger.debug('dfCategoryEvents.columns: %s' %dfCategory[[eventcolumn])
        print('dfCategoryEvents.columns :', dfCategoryEvents.shape)
        # na olan kısımları drop et.
        dfCategoryEvents = dfCategoryEvents[dfCategoryEvents[categorycolumn].notna(
        )]

        self.logger.debug('dfEvent[eventcolumn].unique(): %s' % dfEvent[eventcolumn].unique())
        #print('dfCategoryEvents.columns :', dfCategoryEvents.shape)
        return dfCategoryEvents

    def fault(self, dffault, path):
        Faults_MainCategory = pd.read_excel(path)
        Faults_MainCategory = Faults_MainCategory.drop(
            Faults_MainCategory.columns[0], axis='columns')
        Faults_MainCategory.columns = [
            'Number', 'Braking', 'Main Category', 'Event ID']
        FaultAllCategoryEvents = pd.merge(dffault, Faults_MainCategory[[
                                          'Main Category', 'Event ID']], how='left',  on='Event ID')
        FaultAllCategoryEvents = FaultAllCategoryEvents[FaultAllCategoryEvents['Main Category'].notna(
        )]
        return FaultAllCategoryEvents

    def create_labels_from_targetcolumn(self, tagname, eventscolumn):
        logs_faults = Upload_Download_Pickle().download_pickle(
            self.intermediate_path, 'Logs_Faults'+tagname)

        #####Create Labels#####
        # Event referans alarak once binary sonra multiclass sonra da onehot labellar oluştur.
        # Yeni Labellar oluşturmadan önceki kolon sayısı before_size_df
        default_size = logs_faults.shape[1]
        # Target Column Faults
        faults = logs_faults[logs_faults.columns[range(
            default_size-1, default_size)]]
        # 1.Eventler için Binary Multiclass ve Onehot labellar oluştur.
        logs_faults = self.create_binary_multiclass_onehot_labels(
            logs_faults, eventscolumn)
        # print('Faults',Faults.columns)
        return logs_faults, faults
