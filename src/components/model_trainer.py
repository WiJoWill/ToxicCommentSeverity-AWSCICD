import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import random
import pickle
from torch.cuda.amp import autocast, GradScaler
import time
from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, get_linear_schedule_with_warmup, DebertaTokenizer
from transformers.models.deberta.modeling_deberta import ContextPooler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class JRSDebertaDataset(Dataset):
    def __init__(self, id_list, tokenizer, data_dict, max_len):
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
                                   return_tensors='pt')
        target = self.data_dict[self.id_list[index]]['labels']
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze(), target

class JRSDebertaModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super(JRSDebertaModel, self).__init__(config)
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

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 1001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # prepare input
    with open('data_clean/jigsaw18_train_id_list1.pickle', 'rb') as f:
        id_list = pickle.load(f)
    with open('data_clean/jigsaw18_train_datadict.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    print(len(id_list), len(data_dict))

    # hyperparameters
    learning_rate = 0.00003
    max_len = 384
    batch_size = 32
    num_epoch = 3
    model_path = "microsoft/deberta-base"

    config = DebertaConfig.from_pretrained(model_path)
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSDebertaModel.from_pretrained(model_path, config=config)

    model.to(device)

    train_datagen = JRSDebertaDataset(id_list, tokenizer, data_dict, max_len)
    train_generator = DataLoader(dataset=train_datagen,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8,
                                 pin_memory=torch.cuda.is_available())

    start_time = time.time()

    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_train_steps = int(len(id_list) / (batch_size * 3) * num_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    
    for ep in range(num_epoch):
        losses = AverageMeter()
        model.train()
        for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_target) in enumerate(train_generator):
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            batch_target = torch.from_numpy(np.array(batch_target)).float().to(device)

            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                loss = nn.BCEWithLogitsLoss()(logits, batch_target)

            losses.update(loss.item(), logits.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

    out_dir = 'weights/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, 'weights'))

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
