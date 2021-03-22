"""
    Author: Nober Tai
    github: kksweet
    Purpose: 
        This experiment is aim to test the consistency of classifier
"""

from Classifier.TriFn import TriFn
from Classifier.TriFnPredictor import TriFn_Predictor
from Classifier.TriFn import json2df
import json
import jieba
import pandas as pd


if __name__ == "__main__":

    with open("./data/real_news_2021-02-20_dump.json") as file:
        real_news = json.load(file)

    ls = []
    for news in real_news:
        ls.append([list(jieba.cut(news['title']))])


    df = pd.DataFrame(ls, columns=['token_title'])

    predictor = TriFn_Predictor('cuda', model_dir="./model/model_src-official")

    preds = []
    for i in range(0,len(ls), 1000):
        df = pd.DataFrame(ls[i:i+1000], columns=['token_title'])
        pred = predictor.predict(df)
        preds.extend(pred)

    wrong_news = []
    for news, p in zip(real_news, preds):
        news['pred'] = p
        if news['pred'][0] > news['pred'][1]:
            wrong_news.append(news)
    # json_str = json.dumps(real_news, indent=4, ensure_ascii=False)
    # with open('./data/real_news_result.json', "w", encoding='utf8') as file:
    #     file.write(json_str)
    # file.close()


    json_str = json.dumps(wrong_news, indent=4, ensure_ascii=False)
    with open('./data/wrong_news_result.json', "w", encoding='utf8') as file:
        file.write(json_str)
    file.close()

