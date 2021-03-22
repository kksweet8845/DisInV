try:
    import torch
    from torch.utils.data import DataLoader
except:
    pass
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

class Scores:
    def __init__(self, fn, threshold):
        self.threshold = threshold
        self.n_accuracy = 0
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.name = 'F1'
        self.afn = fn
        self.total = 0

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.tp = 0
        self.tn = 0
        self.n_corrects = 0
        self.total = 0

    def update(self, predicts, groundTruth):
        if self.afn == 'softmax':
            max_v, ids = torch.max(predicts, dim=1, keepdim=True)
            predicts = predicts.masked_fill( predicts != max_v, 0)
            predicts = predicts.masked_fill( predicts == max_v, 1)
        elif self.afn == 'sigmoid':
            predicts = predicts > self.threshold

        self.total += predicts.shape[0]

        tn_fp = torch.sum(predicts, 0)[0].data.item()# tn + fp
        tp_fn = torch.sum(predicts, 0)[1].data.item() # tp + fn

        tn = torch.sum(groundTruth[:, 0].type(torch.uint8) * predicts[:, 0]).data.item() # tn
        tp = torch.sum(groundTruth[:, 1].type(torch.uint8) * predicts[:, 1]).data.item() # tp


        fp = tn_fp - tn
        fn = tp_fn - tp

        self.tn += tn
        self.tp += tp
        self.fp += fp
        self.fn += fn

        self.n_corrects += (tp + tn)


    def accuracy(self):
        return self.n_corrects / (self.total + 1e-20)

    def precision(self):
        t_precision = self.tp / ( self.tp + self.fp + 1e-20)
        f_precision = self.tn / ( self.fn + self.tn + 1e-20)
        return t_precision, f_precision

    def recall(self):
        t_recall = self.tp / ( self.tp + self.fn + 1e-20)
        f_recall = self.tn / ( self.tn + self.fp + 1e-20)
        return t_recall, f_recall

    def f1(self):
        t_precision, f_precision = self.precision()
        t_recall, f_recall = self.recall()

        t_f1 = 2 * ( t_recall * t_precision ) / ( t_recall + t_precision + 1e-20 )
        f_f1 = 2 * ( f_recall * f_precision ) / ( f_recall + f_precision + 1e-20 )

        return t_f1, f_f1
    
        


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
    def __init__(self, num_p, dim, model_dir="./model_src", afn='softmax', latent_space=32):
        super(TriFn, self).__init__()

        # word embedding
        # self.wordEmbedding = WordEmbedding(model_dir)

        self.inter_dim = latent_space
        # News content embedding
        self.D = nn.Linear(dim, self.inter_dim)
        # self.D = nn.GRU(self.wordEmbedding.dim, # input_feature_size
        #                 512, # output_feature_size
        #                 num_layers=32,
        #                 bidirectional=True,
        #                 batch_first=True)
        self.V = nn.Linear(self.inter_dim, dim)
        # self.V = nn.GRU(512,
        #                 self.wordEmbedding.dim,
        #                 num_layers=32,
        #                 bidirectional=True,
        #                 batch_first=True)

        self.B_ = nn.Linear(num_p, dim)
        self.q  = nn.Linear(self.inter_dim, 2)

        self.p  = nn.Linear(self.inter_dim, 2)
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

        p1 = self.D(x) # output ( N, seq-len, 128 )
        p2 = self.p(p1) # output ( N, seq-len, 2 )
        pred = torch.relu(p2) # output ( N, seq-len, 2 )
        if self.afn == 'softmax':
            pred = torch.sum(pred, 1) # output ( N, 2 )
            pred = self.softmax(pred) # output ( N, 2 )

        return x2, o2, pred

        # return x2, pred




class TriFn_Trainer:
    def __init__(self, device, train, model_dir, test=None, pretrained=None, batch_size=32, afn='softmax', lan='chinese', hyper_brand=0.5, latent_space=128):

        self.lan = lan
        self.wordEmbedding = WordEmbedding(model_dir)
        self.model = TriFn(20, self.wordEmbedding.dim, model_dir=model_dir, latent_space=latent_space)
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
        self.model_dir = model_dir
        self.writer = SummaryWriter()

        self.hyper_brand = hyper_brand

    def close(self):
        del self.wordEmbedding
        del self.model
        del self.opt
        del self.criteria
        del self.MSELoss
        del self.writer
        del self.train_data
        del self.test_data

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
            batch_size = len(self.test_data)

        dataLoader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=self.create_mini_batch,
                                num_workers=0)
        
        trange = tqdm(enumerate(dataLoader), total=len(dataLoader), desc=f"{description} {epoch}")

        total_data_length = len(dataset)/self.batch_size

        loss = 0
        acc = 0
        precision = 0
        recall = 0
        _opt = self.opt

        logits_ls = []
        labels_ls = []
        accuracy = 0
        score = Scores('softmax', 0.3)
        for i, (x_ls, standpoint_ts, brand_ts, label_ts) in trange:
            logits, batch_loss, x_ts = self._run_iter(x_ls, standpoint_ts, brand_ts, label_ts)
            if i == 0 and epoch == 0:
                self.sample = (self.wordEmbedding(x_ls).to(self.device), brand_ts.to(self.device))
            _loss = 0
            if training:
                _opt.zero_grad()
                batch_loss.backward(retain_graph=True)
                _opt.step()
            
            score.update(logits, label_ts)

            t_p, f_p = score.precision()
            t_r, f_r = score.recall()
            trange.set_postfix(loss=batch_loss.item(), precision=t_p, recall=t_r)

            loss = (loss * (i / (i+1))) + (batch_loss.item() / (i+1))
            precision = (precision * (i / (i+1))) + (t_p / (i+1))
            recall = (recall * (i / (i+1))) + (t_r / (i+1))

            if training:
                self.history['train'].append({'loss' : loss, 'acc' : score.accuracy() })
                # self.writer.add_scalar(f"Loss/train/epoch{epoch}", loss, i)
                # self.writer.add_scalar(f"Accuracy/train/epoch{epoch}", self.accuracy(logits, label_ts), i)
            
            if not training:
                max_v, ids = torch.max(logits, dim=1, keepdim=True)
                # print(torch.squeeze(ids))
                logits_ls.extend(torch.squeeze(ids))
                max_v, ids = torch.max(label_ts, dim=1, keepdim=True)
                labels_ls.extend(torch.squeeze(ids))
                # accuracy = score.accuracy()

            del x_ls
            del standpoint_ts
            del brand_ts
            del label_ts


        del dataLoader

        if training:
            return loss, precision, recall
        else:
            # max_v, pred_t = torch.max(torch.tensor(logits_ls), dim=1, keepdim=True)
            # pred_t = list(map(lambda x: x.index(1), logits_ls))
            pred_t = logits_ls
            # label_t = list(map(lambda x: x.index(1), labels_ls))
            label_t = labels_ls
            # print(pred_t, label_t)
            return metrics.classification_report(label_t, pred_t, target_names=['0', '1']), pred_t, score

    def _run_iter(self, x_ls, standpoint_ts, brand_ts, label_ts):

        # standpoint_ts = standpoint_ts.to(self.device)
        brand_ts = brand_ts.to(self.device)

        x_ts = self.wordEmbedding(x_ls)
        x_ts = x_ts.to(self.device)
        x, o, pred = self.model(x_ts, brand_ts)
        # x, pred = self.model(x_ts, brand_ts)

        loss_x = self.MSELoss(x.cpu(), x_ts.cpu())
        loss_brand = self.criteria(o.cpu(), standpoint_ts.cpu())
        loss_pred = self.criteria(pred.cpu(), label_ts.cpu())

        loss = loss_x  + loss_pred  + self.hyper_brand * loss_brand

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
            loss, precision, recall = self._run_epoch(i, True)
            self.writer.add_scalar(f"{version}/Loss/train/", loss, i)
            self.writer.add_scalar(f"{version}/Accuracy/train", precision, i)
            self.writer.add_scalar(f"{version}Recall/train", recall, i)
            # self.save(i, version)
        
        torch.save(self.model.state_dict(), path.join(self.model_dir, f'pytorch_model.bin'))
        self.writer.add_graph(self.model, self.sample)
        self.writer.close()

        return self.validate()

    def validate(self):
        report, pred, score = self._run_epoch(1, False)

        print(report)
        # print(f"accuracy : {acc}")
        t_p, f_p = score.precision()
        t_r, f_r = score.recall()
        t_f1, f_f1 = score.f1()

        if t_p < 0.5 or t_r < 0.5:
            print(f"Validating report, Precision : {t_p}, Recall : {t_r}")

        return t_p, f_p, t_r, f_r, t_f1, f_f1

    def predict(self, dataset):


        dataLoader = DataLoader(dataset=dataset,
                                batch_size=len(dataset),
                                shuffle=False,
                                collate_fn=self.create_mini_batch,
                                num_workers=0)
        
        trange = tqdm(enumerate(dataLoader), total=len(dataLoader))


        logits_ls = []
        labels_ls = []
        score = Scores('softmax', 0.3)
        for i, (x_ls, standpoint_ts, brand_ts, label_ts) in trange:
            logits, batch_loss, x_ts = self._run_iter(x_ls, standpoint_ts, brand_ts, label_ts)
            
            score.update(logits, label_ts)
            max_v, ids = torch.max(logits, dim=1, keepdim=True)
            # print(torch.squeeze(ids))
            logits_ls.extend(torch.squeeze(ids))
            max_v, ids = torch.max(label_ts, dim=1, keepdim=True)
            labels_ls.extend(torch.squeeze(ids))
                # accuracy = score.accuracy()

        
        # max_v, pred_t = torch.max(torch.tensor(logits_ls), dim=1, keepdim=True)
        # pred_t = list(map(lambda x: x.index(1), logits_ls))
        pred_t = logits_ls
        # label_t = list(map(lambda x: x.index(1), labels_ls))
        label_t = labels_ls

        # print(pred_t, label_t)
        return metrics.classification_report(label_t, pred_t, target_names=['0', '1']), pred_t, score




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



# class TriFn_Predicter:
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
        try:
            processed['brand_id'] = sample['brand_id']
        except KeyError:
            pass
        try:
            processed['label'] = sample['label']
        except KeyError:
            pass
        try:
            processed['standpoint'] = sample['standpoint']
        except KeyError:
            pass

        return processed

    def process_dataset(self, dataset):

        processed = []

        for sample in dataset.iterrows():
            processed.append(self.process_sample(sample[1]))

        return processed




# Json to csv
def json2df(json_obj):

    ls = []

    keys = json_obj[0].keys()
    col = keys

    for row in json_obj:
        tmp = []
        for key in keys:
            tmp.append(row[key])
        ls.append(tmp)
        # ls.append([
        #     row['title'],
        #     row['content'],
        #     row['author'],
        #     row['brand_id'],
        #     row['date'],
        #     row['url'],
        #     row['standpoint'],
        #     row['token_title']
        # ])

    # col = [
    #         'title',
    #         'content',
    #         'author',
    #         'brand_id',
    #         'date',
    #         'url',
    #         'standpoint',
    #         'token_title'
    #     ]

    return pd.DataFrame(ls, columns=col)



from sklearn.model_selection import train_test_split



def k_CV(df, desc="", model_dir='./model_src_kv', hyper_brand=0.5, latent_space=128):

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=4)

    # Split
    k = 5
    _bin = int(len(train_df) / 5)
    print(_bin)
    split_df = [  train_df.iloc[i:i+_bin, :]  for i in range(0, len(train_df), _bin) ]

    score_list = {
        't_p' : [],
        'f_p' : [],
        't_r' : [],
        'f_r' : [],
        't_f1' : [],
        'f_f1' : [],
        'accuracy' : []
    }

    test_data = TriFn_dataset(test_df)
    for i in range(k):
        # train
        training_dataset = pd.concat([ split_df[d] for d in range(k) if i != d  ], ignore_index=True)
        validating_dataset = split_df[i]

        train_data = TriFn_dataset(training_dataset)
        valid_data = TriFn_dataset(validating_dataset)


        # print(training_dataset)
        # print(len(training_dataset))
        # print(len(validating_dataset))

        trainer = TriFn_Trainer('cuda', train_data, model_dir, pretrained=False, test=valid_data, batch_size=64, hyper_brand=hyper_brand, latent_space=latent_space)
        
        trainer.train(10, f'TriFn_k{i}_{desc}') 
        
        report, pred_t, score = trainer.predict(test_data)
        t_p, f_p = score.precision()
        t_r, f_r = score.recall()
        t_f1, f_f1 = score.f1()
        acc = score.accuracy()

        score_list['t_p'].append(t_p)
        score_list['f_p'].append(f_p)
        score_list['t_r'].append(t_r)
        score_list['f_r'].append(f_r)
        score_list['t_f1'].append(t_f1)
        score_list['f_f1'].append(f_f1)
        score_list['accuracy'].append(acc)

        trainer.close()

        del trainer
        del train_data
        del valid_data

    key_list = ['t_p', 'f_p', 't_r', 'f_r', 't_f1', 'f_f1', 'accuracy']

    ave_list = [ np.mean(score_list[key]) for key in key_list ]
    
    print("="*30)
    print("Final Testing result")
    _string = ''
    for i, key in enumerate(key_list):
        _string += f"{key} : \n"
        for d in score_list[key]:
            _string += f"{d:3.3f}, "
        _string += "\n\n"
        _string += "-"*30
        _string += "\n\n"

    print("Each scores")
    print(_string)
    _string = ''
    for i, key in enumerate(key_list):
        _string += f"{key} : {ave_list[i]:3.3f}\n"
    print("Average score")
    print(_string)








    



        




# if __name__ == "__main__":

#     # we = WordEmbedding('./model_src')

#     # ls = [['餐廳', '韓國瑜'],
#     #       ['餐廳', '馬英九']]

#     # print(we(np.array(ls)))

#     with open("./data/real_news_token_data.json") as file:
#         real_news = json.load(file)

#     real_news_df = json2df(real_news)
#     real_news_df['label'] = [1] * len(real_news_df)

#     with open("./data/fake_news_token_data.json") as file:
#         fake_news = json.load(file)

#     fake_news_df = json2df(fake_news)
#     fake_news_df['label'] = [0] * len(fake_news_df)


#     compact_df = pd.concat([real_news_df, fake_news_df], ignore_index=True)

#     train_df, test_df = train_test_split(compact_df, test_size=0.2, random_state=2)

#     # train_data = TriFn_dataset(compact_df)
#     train_data = TriFn_dataset(train_df)
#     test_data = TriFn_dataset(test_df)

#     train = True

#     if train == True:
#         trainer = TriFn_Trainer('cuda', train_data, './model_src', pretrained=False, test=test_data, batch_size=64)
#         trainer.train(32, 'TriFn_v1')
#         # trainer.predict()
#     elif train == False:
#         predictor = TriFn_Trainer('cuda', train_data, './model_src', pretrained=True, test=test_data, batch_size=64)
#         # predictor.predict()

#     # K-fold CV

#     k_fold_validating = False

#     if k_fold_validating:
#         k_CV(compact_df)
    







    