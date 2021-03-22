from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from .TriFn import  TriFn_dataset, WordEmbedding, TriFn
from os import path
try:
    import torch
    from torch.utils.data import DataLoader
except:
    pass

import numpy as np


class TriFn_Predictor:
    def __init__(self, device, model_dir="./model_src", afn="softmax"):
        self.wordEmbedding = WordEmbedding(model_dir)
        self.model = TriFn(20, self.wordEmbedding.dim, model_dir=model_dir, latent_space=128)
        self.model.load_state_dict(torch.load(path.join(model_dir, 'pytorch_model.bin')))
        self.model.eval()
        self.model.to(device)

        self.device = device
        self.batch_size = 1
        self.afn = afn
        self.model_dir = model_dir
        self.writer = SummaryWriter()

    def predict(self, df):

        dataset = TriFn_dataset(df)

        dataLoader = DataLoader(dataset=dataset,
                                batch_size=len(dataset),
                                shuffle=False,
                                collate_fn=self.create_mini_batch,
                                num_workers=0)

        trange = tqdm(enumerate(dataLoader), total=len(dataLoader))

        logits_ls = []

        for i, (x_ls, standpoint_ts, brand_ts) in trange:
            pred = self._run_iter(x_ls, standpoint_ts, brand_ts)
            logits_ls.extend(torch.squeeze(pred.detach()).tolist())

        return logits_ls
        

    def _run_iter(self, x_ls, standpoints_ts, brand_ts):
        brand_ts = brand_ts.to(self.device)

        x_ts = self.wordEmbedding(x_ls)
        x_ts = x_ts.to(self.device)

        _, __, pred = self.model(x_ts, brand_ts)

        return pred.cpu()



    def create_mini_batch(self, samples):
        """
            {
                "title" : the content,
                "brand_id" : the brand_id,
            }
            label needs to be one hot formation
            standpoint is also needs to be one hot formation
            brand_id needs to be one hot formation
        """

        standpoint_ls = []
        brand_ls = []
        x_ls = []

        max_len = max([len(x['title']) for x in samples])

        for sample in samples:
            brand_ls.append(0)
            padding_ls = sample['title']
            padding_ls.extend(['</s>']* (max_len - len(sample['title'])) )
            x_ls.append(padding_ls)

        
        batch_size = len(samples)
        standpoint_ts = torch.tensor(standpoint_ls, dtype=torch.long)[..., None]
        brand_ts = torch.tensor(brand_ls, dtype=torch.long)[..., None]
        # label_ts = torch.tensor(label_ls, dtype=torch.long)[..., None]


        standpoint_ts = torch.zeros(batch_size, 2).scatter_(1, standpoint_ts, 1)
        brand_ts = torch.zeros(batch_size, 20).scatter_(1, brand_ts, 1)
        # label_ts = torch.zeros(batch_size, 2).scatter_(1, label_ts, 1)

        return np.array(x_ls), standpoint_ts, brand_ts

        



import json
from .TriFn import json2df
import pandas as pd
if __name__ == "__main__":

    # for test purpose
    with open("./data/real_news_token_data.json") as file:
        real_news = json.load(file)

    real_news_df = json2df(real_news)
    
    with open("./data/fake_news_token_data.json") as file:
        fake_news = json.load(file)

    fake_news_df = json2df(fake_news)

    compact_df = pd.concat([real_news_df, fake_news_df], ignore_index=True).iloc[:100, :]

    predictor = TriFn_Predictor('cuda')

    pred = predictor.predict(compact_df)

    print(pred)


