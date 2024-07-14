import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    validation_path: str = os.path.join('datasource', "validation_data.csv")
    train_datasource_18_path: str = os.path.join('datasource',"jigsaw18_train.csv")
    train_datasource_19_path: str = os.path.join('datasource',"jigsaw19_train.csv")
    train_datasource_ruddit_path: str = os.path.join('datasource',"ruddit_train.csv")

    train_18_idlist: str = os.path.join('data_clean', "jigsaw18_train_id_list.pickle")
    train_18_idlist1: str = os.path.join('data_clean', "jigsaw18_train_id_list1.pickle") # not in validation dataset
    train_18_datadict: str = os.path.join('data_clean', "jigsaw18_train_datadict.pickle")

    train_19_idlist: str = os.path.join('data_clean', "jigsaw19_train_id_list.pickle")
    train_19_datadict: str = os.path.join('data_clean', "jigsaw19_train_datadict.pickle")

    
    ruddit_idlist: str = os.path.join('data_clean', "ruddit_train_id_list.pickle")
    ruddit_datadict: str = os.path.join('data_clean', "ruddit_train_datadict.pickle")

    test_data_path: str = os.path.join('datasource',"test.csv")
    raw_data_path: str = os.path.join('datasource',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.ingestion_config.validation_path)
            more_toxic_list = df['more_toxic'].values
            less_toxic_list = df['less_toxic'].values
            toxic_set = set(list(more_toxic_list) + list(less_toxic_list))
            toxic_list = sorted(list(toxic_set))
            # print(len(more_toxic_list), len(less_toxic_list), len(toxic_list))
            
            logging.info('Read the jigswa18 train files as dataframe')
            df_18 = pd.read_csv(self.ingestion_config.train_datasource_18_path, encoding='latin1')

            id_list = df_18['id'].values
            id_list1 = []
            comment_list = df_18['comment_text'].values
            toxic_list = df_18['toxic'].values
            severe_toxic_list = df_18['severe_toxic'].values
            obscene_list = df_18['obscene'].values
            threat_list = df_18['threat'].values
            insult_list = df_18['insult'].values
            identity_hate_list = df_18['identity_hate'].values
            data_dict = {}
            for i in tqdm(range(len(id_list))):
                data_dict[id_list[i]] = {'text': comment_list[i], 'labels': np.array([toxic_list[i], severe_toxic_list[i], obscene_list[i], threat_list[i], insult_list[i], identity_hate_list[i]])}
                if comment_list[i] not in toxic_set:
                    id_list1.append(id_list[i])
            print(len(id_list), len(id_list1))

            os.makedirs(os.path.dirname(self.ingestion_config.train_18_idlist),exist_ok=True)
            with open(self.ingestion_config.train_18_idlist, 'wb') as f:
                pickle.dump(id_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.ingestion_config.train_18_idlist1, 'wb') as f:
                pickle.dump(id_list1, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.ingestion_config.train_18_datadict, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            logging.info("Ingestion of the jigswa18 data is completed")

            df1 = pd.read_csv(self.ingestion_config.train_datasource_19_path, encoding='latin1')
            id_list = df1['id'].values
            id_list1 = []
            comment_list = df1['comment_text'].values
            toxic_list = df1['toxicity'].values
            severe_toxic_list = df1['severe_toxicity'].values
            obscene_list = df1['obscene'].values
            threat_list = df1['threat'].values
            insult_list = df1['insult'].values
            identity_hate_list = df1['identity_attack'].values
            sexual_list = df1['sexual_explicit'].values
            data_dict = {}
            for i in tqdm(range(len(id_list))):
                if isinstance(comment_list[i], str):
                    data_dict[id_list[i]] = {'text': comment_list[i], 'labels': np.array([toxic_list[i], severe_toxic_list[i], obscene_list[i], threat_list[i], insult_list[i], identity_hate_list[i], sexual_list[i]])}
                    id_list1.append(id_list[i])
            print(len(id_list), len(id_list1))

            os.makedirs(os.path.dirname(self.ingestion_config.train_19_idlist),exist_ok=True)
            with open(self.ingestion_config.train_19_idlist, 'wb') as f:
                pickle.dump(id_list1, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.ingestion_config.train_19_datadict, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            df1 = pd.read_csv(self.ingestion_config.train_datasource_ruddit_path, encoding='latin1')
            comment_list = df1['txt'].values
            id_list = df1['id'].values
            id_list1 = []
            offensiveness_score_list = (df1["offensiveness_score"].values + 1.) / 2.
            print(offensiveness_score_list.mean(), offensiveness_score_list.min(), offensiveness_score_list.max())
            data_dict = {}
            for i in tqdm(range(len(id_list))):
                data_dict[id_list[i]] = {'text': comment_list[i], 'labels': offensiveness_score_list[i]}
                if isinstance(comment_list[i], str) and comment_list[i] != '[deleted]':
                    id_list1.append(id_list[i])
            print(len(id_list), len(id_list1))

            os.makedirs(os.path.dirname(self.ingestion_config.ruddit_idlist),exist_ok=True)
            with open(self.ingestion_config.ruddit_idlist, 'wb') as f:
                pickle.dump(id_list1, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.ingestion_config.ruddit_datadict, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            pass
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

    # data_transformation = DataTransformation()
    # train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)
    '''
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    '''