import sys
import pandas as pd
import numpy as np
from src.utils.utils_p import YamlParser, Config_Paths, Dataset_Configs, Upload_Download_Pickle, CombineDataset_Configs
from src.intermediate.Combine_Datasets import Combine_Datasets
sys.path.append('..')


class Combine():

    cd = Combine_Datasets()
    # 1.Create A List of Datasets from Each Tag:
    features_list = cd.combine_data('Features')
    labels_list = cd.combine_data('Labels')

    # 2.Create DataFrames from List:
    features_df = cd.create_df_from_list(features_list)
    labels_df = cd.create_df_from_list(labels_list)

    # 3.Optional: Only Select Datasets has enough Target Examples & :
    cd.filter_tag_target_list(features_list, labels_list)
    cd.filter_tag_target_df(features_df, labels_df)



