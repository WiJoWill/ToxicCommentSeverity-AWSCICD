import sys
import pandas as pd
from src.exception import CustomException
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from torch.cuda.amp import autocast
import time
from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, DebertaTokenizer
from transformers.models.deberta.modeling_deberta import ContextPooler
from transformers import DebertaTokenizer, DebertaConfig, DebertaForSequenceClassification




class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

class JRSModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super(JRSModel, self).__init__(config)
        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = nn.Linear(output_dim, 6)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        logits = self.classifier(pooled_output)
        return logits

def get_deberta_base1(text_list, weight_list):
    start_time = time.time()

    # parameters
    max_len = 192
    batch_size = 8

    # build model
    toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts/')
    config_path = os.path.join(model_path, 'config.json')
    tokenizer = DebertaTokenizer.from_pretrained(model_path)

    # Resolve the absolute paths
    model_path = os.path.abspath(os.path.join(model_path, 'model'))
    config_path = os.path.abspath(config_path)

    config = DebertaConfig.from_pretrained(config_path)
    model = JRSModel.from_pretrained(model_path, config=config)

    # Check if GPU is available and move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j * batch_size
            end = start + batch_size
            if j == len(generator) - 1:
                end = len(generator.dataset)
            
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)

            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(6):
        toxic_score += toxic_pred[:,i] * weight_list[i]
    ranks = toxic_score.argsort().argsort()
    # print(toxic_score[:20])
    # print(ranks[:20])

    end_time = time.time()
    print(end_time - start_time)

    return toxic_pred, toxic_score

class PredictPipeline:
    def __init__(self):
        self.weight_list = [20.0, 18.0, 4.0, 4.0, 1.0, 4.0]

    
    def predict(self, text_list):
        try:
            start_time = time.time()

            # parameters
            max_len = 192
            batch_size = 8

            # build model
            toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
            
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts/')
            config_path = os.path.join(model_path, 'config.json')
            tokenizer = DebertaTokenizer.from_pretrained(model_path)
            print('Tokenizer has been processed:{}'.format(model_path))

            # Resolve the absolute paths
            model_path = os.path.abspath(os.path.join(model_path, 'model'))
            config_path = os.path.abspath(config_path)
            print('Model Path is {}'.format(model_path))

            config = DebertaConfig.from_pretrained(config_path)
            print('Config has been processed:{}'.format(config_path))

            model = JRSModel.from_pretrained(model_path, config=config)
            print('Model has been processed:{}'.format(model_path))

            # Check if GPU is available and move model to the appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            # iterator for validation
            dataset = JRSDataset(text_list, tokenizer, max_len)
            print('Dataset worked')
            generator = DataLoader(dataset=dataset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                num_workers=0, 
                                pin_memory=False)
            print('dataloader worked')
            
            for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
                with torch.no_grad():
                    start = j * batch_size
                    end = start + batch_size
                    if j == len(generator) - 1:
                        end = len(generator.dataset)
                    
                    batch_input_ids = batch_input_ids.to(device)
                    batch_attention_mask = batch_attention_mask.to(device)
                    batch_token_type_ids = batch_token_type_ids.to(device)

                    with autocast():
                        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                    toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()
            print('Prediction Finished')
            ### result
            toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
            for i in range(6):
                toxic_score += toxic_pred[:,i] * self.weight_list[i]
            # ranks = toxic_score.argsort().argsort()
            # print(toxic_score[:20])
            # print(ranks[:20])

            end_time = time.time()
            print(end_time - start_time)

            return toxic_pred, toxic_score
        
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(self, text: str):
        self.text = text

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "text": [self.text],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
     # df_path = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts/', 'comments_to_score.csv')
    # df = pd.read_csv(df_path)
    text_list = ['Not good. This is pretty awful and disgusting.']
    weight_list3 = [20.0, 18.0, 4.0, 4.0, 1.0, 4.0]
    predict_pipeline = PredictPipeline()
    toxic_pred, toxic_score = predict_pipeline.predict(text_list)
    print(toxic_pred, toxic_score)

