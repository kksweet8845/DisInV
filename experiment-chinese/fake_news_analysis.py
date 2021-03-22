import json
import numpy as np
from Classifier.TriFn import Scores
import torch

class Publisher_record(Scores):
    def __init__(self, brand_id):
        super().__init__('softmax', 0.3)
        self.brand_id = brand_id

    def report(self):
        t_r, f_r = self.recall()
        t_p, f_p = self.precision()
        t_f1, f_f1 = self.f1()
        print("=" * 30)
        print("Brand_id : {}".format(self.brand_id))
        print("confusion matrix : tp : {}, tn : {}, fp : {}, fn : {}\n".format(self.tp, self.tn, self.fp, self.fn))
        print("Recall \nt_r : {}, f_r : {}\n".format(t_r, f_r))
        print("Precision \nt_p : {}, f_p : {}\n".format(t_p, f_p))
        print("f1 score \n t_f1 : {}, f_f1 : {}\n".format(t_f1, f_f1))


if __name__ == "__main__":

    ## Count the report of prediction -- real

    with open("./data/fake_news_result.json") as file:
        fake_news = json.load(file)

    ## calculate the total number of news
    publisher = {}

    # truth_gt = [ [0, 1] for i in range(len(real_news))  ]

    # false_gt = [ [1, 0] for i in range()]

    for news in fake_news:
        if news['brand_id'] in publisher.keys():
            publisher[news['brand_id']].append(news['pred'])
        else:
            publisher[news['brand_id']] = []
            publisher[news['brand_id']].append(news['pred'])

    # score_ls = []
    for brand_id in publisher.keys():
        tmp = Publisher_record(brand_id)
        truth_gt = [ [1, 0] for i in range(len(publisher[brand_id])) ]
        tmp.update(torch.tensor(publisher[brand_id]), torch.tensor(truth_gt))
        tmp.report()
        # score_ls.append(tmp)


    





    





