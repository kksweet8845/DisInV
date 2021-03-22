"""
    Author: Nober Tai
    github: kksweet
    Purpose:
        This experiment is aimed to test the consistency of classifier
"""

from Classifier.TriFn import TriFn
from Classifier.TriFnPredictor import TriFn_Predictor
import json
import jieba
import pandas as pd



if __name__ == "__main__":

    with open("./data/fake_news_2021-02-20_dump.json") as file:
        fake_news = json.load(file)

    ls = []
    for news in fake_news:
        ls.append([list(jieba.cut(news['title']))])

    df = pd.DataFrame(ls, columns=['token_title'])

    predictor = TriFn_Predictor('cuda', model_dir="./model_src-official")

    preds = []
    for i in range(0, len(ls), 1000):
        df = pd.DataFrame(ls[i:i+1000], columns=['token_title'])
        pred = predictor.predict(df)
        preds.extend(pred)

    for news, p in zip(fake_news, preds):
        news['pred'] = p


    json_str = json.dumps(fake_news, indent=4, ensure_ascii=False)
    with open("./data/fake_news_result.json", "w", encoding='utf8') as file:
        file.write(json_str)
    file.close()
