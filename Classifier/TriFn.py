try:
    import torch
    from torch.utils.data import DataLoader
except:
    pass
from utils import vec2Tensor
from os import path
import pickle
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

class WordEmbedding:
    def __init__(self, model_dir):
        with open(path.join(model_dir, "embedding.ts"), 'rb') as file:
            self.embedding_ts = pickle.load(file)

        with open(path.join(model_dir, "lookup.tb"), 'rb') as file:
            self.lookup_tb = pickle.load(file)

        # print(self.lookup_tb)
        self.vocab, self.dim = self.embedding_ts.shape
        self.embedding = nn.Embedding.from_pretrained(self.embedding_ts)

    def __call__(self, arr):
    
        def f(s):
            try:
                return self.lookup_tb[s]
            except KeyError:
                return self.lookup_tb['</s>']
        
        vf = np.vectorize(f)
        t = vf(arr)

        return self.embedding(torch.tensor(t, dtype=torch.long))
    
    
class TriFn(nn.Module):
    def __init__(self, num_p, dim, model_dir="./model_src", afn='softmax'):
        super(TriFn, self).__init__()

        # word embedding
        # self.wordEmbedding = WordEmbedding(model_dir)

        # News content embedding
        self.D = nn.Linear(dim, 128)
        # self.D = nn.GRU(self.wordEmbedding.dim, # input_feature_size
        #                 512, # output_feature_size
        #                 num_layers=32,
        #                 bidirectional=True,
        #                 batch_first=True)
        self.V = nn.Linear(128, dim)
        # self.V = nn.GRU(512,
        #                 self.wordEmbedding.dim,
        #                 num_layers=32,
        #                 bidirectional=True,
        #                 batch_first=True)

        self.B_ = nn.Linear(num_p, dim)
        self.q  = nn.Linear(128, 2)

        self.p  = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)
        self.afn = afn

    def forward(self, x, h_0):
        """
            raw_x : (batch, seq_len, dim)
        """
        # x = self.wordEmbedding(raw_x)
        x1 = self.D(x)
        x2 = self.V(x1)

        o = self.B_(h_0)
        o1 = self.D(o)
        o2 = self.q(o1)
        o2 = torch.relu(o2)
        o2 = self.softmax(o2)

        p1 = self.D(x)
        p2 = self.p(p1)
        pred = torch.relu(p2)
        if self.afn == 'softmax':
            pred = self.softmax(pred)

        return x2, o2, pred[:, 0]


class TriFn_Trainer:
    def __init__(self, device, train, model_dir, test=None, pretrained=None, batch_size=32, afn='softmax'):


        self.wordEmbedding = WordEmbedding(model_dir)
        self.model = TriFn(20, self.wordEmbedding.dim, model_dir=model_dir)
        if pretrained == True:
            self.model.load_state_dict(torch.load(path.join(model_dir, 'pytorch_model.bin')))
            self.model.eval()

        self.opt = torch.optim.Adam(self.model.parameters())
        self.criteria = nn.BCELoss()
        self.MSELoss = nn.MSELoss()
        self.model.to(device)
        self.device = device
        self.train_data = train
        self.test_data = test
        self.batch_size = batch_size
        self.history = {'train' : []}
        self.afn = afn
        self.model_dir=model_dir
        self.writer = SummaryWriter()

    def accuracy(self, predicts, gt):
        max_v, ids = torch.max(predicts, dim=1, keepdim=True)
        predicts = predicts.masked_fill(predicts != max_v, 0)
        predicts = predicts.masked_fill(predicts == max_v, 1)

        precision = torch.sum(predicts).data.item()
        recall = torch.sum(gt).data.item()
        n_correct = torch.sum(gt.type(torch.uint8) * predicts).data.item()

        return n_correct / (precision + 1e-20), n_correct / recall

    def _run_epoch(self, epoch, training):

        self.model.train(training)
        if training:
            description = 'Train'
            dataset = self.train_data
            shuffle = True
            batch_size=self.batch_size
        else:
            assert epoch == 1, "Not one epoch"
            description = 'Evaluation'
            dataset = self.test_data
            shuffle = False
            batch_size= len(self.test_data)
            
        
        dataLoader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=self.create_mini_batch,
                                num_workers=4)
        
        trange = tqdm(enumerate(dataLoader), total=len(dataLoader), desc=f"{description} {epoch}")

        total_data_length = len(dataset)/self.batch_size

        loss = 0
        acc = 0
        precision = 0
        recall = 0
        _opt = self.opt

        logits_ls = []
        labels_ls = []
        for i, (x_ls, standpoint_ts, brand_ts, label_ts) in trange:
            logits, batch_loss, x_ts = self._run_iter(x_ls, standpoint_ts, brand_ts, label_ts)
            if i == 0 and epoch == 0:
                self.sample = (self.wordEmbedding(x_ls).to(self.device), brand_ts.to(self.device))
            _loss = 0
            if training:
                _opt.zero_grad()
                batch_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
                _opt.step()
            
            _precision, _recall = self.accuracy(logits, label_ts)
            trange.set_postfix(loss=batch_loss.item(), precision=_precision, recall=_recall)

            loss = (loss * (i / (i+1))) + (batch_loss.item() / (i+1))
            precision = (precision * (i / (i+1))) + (_precision / (i+1))
            recall = (recall * (i / (i+1))) + (_recall / (i+1))

            if training:
                self.history['train'].append({'loss' : loss, 'acc' : self.accuracy(logits, label_ts)})
                # self.writer.add_scalar(f"Loss/train/epoch{epoch}", loss, i)
                # self.writer.add_scalar(f"Accuracy/train/epoch{epoch}", self.accuracy(logits, label_ts), i)
            if not training:
                logits_ls.extend(logits)
                labels_ls.extend(label_ts)

        if training:
            return loss, precision, recall
        else:
            pred_t = map(lambda x: x.index(1), logits_ls)
            label_t = map(lambda x: x.index(1), labels_ls)
            return metrics.classification_report(label_t, pred_t, target_names=[0, 1]), pred_t

    def _run_iter(self, x_ls, standpoint_ts, brand_ts, label_ts):

        # standpoint_ts = standpoint_ts.to(self.device)
        brand_ts = brand_ts.to(self.device)

        x_ts = self.wordEmbedding(x_ls)
        x_ts = x_ts.to(self.device)
        x, o, pred = self.model(x_ts, brand_ts)

        loss_x = self.MSELoss(x.cpu(), x_ts.cpu())
        loss_brand = self.criteria(o.cpu(), standpoint_ts.cpu())
        loss_pred = self.criteria(pred.cpu(), label_ts.cpu())

        loss = loss_x + loss_brand + loss_pred

        return pred.cpu(), loss, x_ts

    def save(self, epoch, version):

        folder_path = path.join(self.model_dir, f'model_TriFn_{epoch}_v_{version}')
        
        if not path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.model.state_dict(), path.join(folder_path, f'pytoch_model.bin'))

        json_str = json.dumps(self.history, indent=4, ensure_ascii=False)
        with open(path.join(folder_path, 'history.log'), 'w+') as file:
            file.write(json_str)

    def train(self, epoch, version):

        for i in range(epoch):
            loss, precision, recall= self._run_epoch(i, True)
            self.writer.add_scalar(f"Loss/train", loss, i)
            self.writer.add_scalar(f"Accuracy/train", precision, i)
            self.writer.add_scalar(f"Recall/train", recall, i)
            # self.save(i, version)
        
        torch.save(self.model.state_dict(), path.join(self.model_dir, f'pytorch_model.bin'))

        self.writer.add_graph(self.model, self.sample)
        self.writer.close()

    def predict(self):

        report, pred = self._run_epoch(1, False)

        print(pred)
        print(report)

        

    def create_mini_batch(self, samples):
        """
            {
                "title" : the content,
                "brand_id" : the brand_id,
                "label" : real or fake
                "standpoint" : the standpoint of the x
            }
            label needs to be one hot formation
            standpoint is also needs to be one hot formation
            brand_id needs to be one hot formation
        """

        standpoint_ls = []
        brand_ls = []
        label_ls = []
        x_ls = []

        max_len = max([len(x['title']) for x in samples])

        for sample in samples:
            standpoint_ls.append(sample['standpoint'])
            brand_ls.append(sample['brand_id'])
            label_ls.append(sample['label'])
            padding_ls = sample['title']
            padding_ls.extend(['</s>']* (max_len - len(sample['title'])) )
            x_ls.append(padding_ls)

        
        batch_size = len(samples)
        standpoint_ts = torch.tensor(standpoint_ls, dtype=torch.long)[..., None]
        brand_ts = torch.tensor(brand_ls, dtype=torch.long)[..., None]
        label_ts = torch.tensor(label_ls, dtype=torch.long)[..., None]


        standpoint_ts = torch.zeros(batch_size, 2).scatter_(1, standpoint_ts, 1)
        brand_ts = torch.zeros(batch_size, 20).scatter_(1, brand_ts, 1)
        label_ts = torch.zeros(batch_size, 2).scatter_(1, label_ts, 1)

        return np.array(x_ls), standpoint_ts, brand_ts, label_ts



class TriFn_Predicter:
    
from torch.utils.data import Dataset

class TriFn_dataset(Dataset):

    def __init__(self, dataset, n_workers=4):
        self.raw_dataset = dataset
        self.length = len(dataset)
        self.processed_dataset = self.process_dataset(dataset)

    
    def __getitem__(self, idx):
        return self.processed_dataset[idx]

    def __len__(self):
        return self.length

    def process_sample(self, sample):
        """ """
        processed = {}
        processed['title'] = sample['token_title']
        processed['brand_id'] = sample['brand_id']
        processed['label'] = sample['label']
        processed['standpoint'] = sample['standpoint']

        return processed

    def process_dataset(self, dataset):

        processed = []

        for sample in dataset.iterrows():
            processed.append(self.process_sample(sample[1]))

        return processed




# Json to csv
def json2df(json_obj):

    ls = []
    for row in json_obj:
        ls.append([
            row['title'],
            row['content'],
            row['author'],
            row['brand_id'],
            row['date'],
            row['url'],
            row['standpoint'],
            row['token_title']
        ])

    col = [
            'title',
            'content',
            'author',
            'brand_id',
            'date',
            'url',
            'standpoint',
            'token_title'
        ]

    return pd.DataFrame(ls, columns=col)

if __name__ == "__main__":

    # we = WordEmbedding('./model_src')

    # ls = [['餐廳', '韓國瑜'],
    #       ['餐廳', '馬英九']]

    # print(we(np.array(ls)))

    with open("./data/real_news_token_data.json") as file:
        real_news = json.load(file)

    real_news_df = json2df(real_news)
    real_news_df['label'] = [1] * len(real_news_df)

    with open("./data/fake_news_token_data.json") as file:
        fake_news = json.load(file)

    fake_news_df = json2df(fake_news)
    fake_news_df['label'] = [0] * len(fake_news_df)


    compact_df = pd.concat([real_news_df, fake_news_df], ignore_index=True)

    train_data = TriFn_dataset(compact_df)

    trainer = TriFn_Trainer('cuda', train_data, './model_src', batch_size=64)

    # trainer.train(100, 'TriFn_v1')
    trainer.predict()






    