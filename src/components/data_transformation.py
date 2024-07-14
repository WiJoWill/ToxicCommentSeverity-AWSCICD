import sys
from dataclasses import dataclass
import pickle
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from torch.utils.data import Dataset, DataLoader

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object, load_object

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class JRSDataDataset(Dataset):
    def __init__(self, id_list, tokenizer_name, data_dict, max_len):
        self.id_list = id_list
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.data_dict = data_dict
        self.max_len = max_len

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        tokenized = self.tokenizer(text=self.data_dict[self.id_list[index]]['text'],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_len,
                                   return_tensors='pt')
        target = self.data_dict[self.id_list[index]]['labels']
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze(), target

class JRSDebertaDataset(Dataset):
    def __init__(self, id_list, tokenizer, data_dict, max_len):
        self.id_list=id_list
        self.tokenizer=tokenizer
        self.data_dict=data_dict
        self.max_len=max_len
    def __len__(self):
        return len(self.id_list)
    def __getitem__(self, index):
        tokenized = self.tokenizer(text=self.data_dict[self.id_list[index]]['text'],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_len,
                                   return_tensors='pt')
        target = self.data_dict[self.id_list[index]]['labels']
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze(), target


if __name__ == "__main__":
    id_list, data_dict = load_object('data_clean\jigsaw18_train_id_list.pickle'), load_object('data_clean\jigsaw18_train_datadict.pickle')
    tokenizer_name = "microsoft/deberta-base"
    dataset = JRSDataDataset(id_list, tokenizer_name, data_dict, max_len=384)
    print("Dataset length:", len(dataset))

'''
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('datasource',"proprocessor.pkl")

class DataTransformation(Dataset):
    def __init__(self, id_list, tokenizer, data_dict, max_len = 256):
        self.data_transformation_config = DataTransformationConfig()
        self.id_list = id_list
        self.tokenizer = tokenizer
        self.data_dict = data_dict
        self.max_len = max_len

    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, index):
        tokenized = self.tokenizer(text=self.data_dict[self.id_list[index]]['text'],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_len,
                                   return_attention_mask=True,
                                   return_token_type_ids=True,
                                   return_tensors='pt')
        target = self.data_dict[self.id_list[index]]['labels']
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), \
            tokenized['token_type_ids'].squeeze(), target
'''