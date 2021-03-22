import json
from Classifier.TriFn import json2df, k_CV
from sklearn.model_selection import train_test_split
import pandas as pd



if __name__ == "__main__":


    with open("./data/eng/real_news.json") as file:
        real_news = json.load(file)

    real_news_df = json2df(real_news)


    with open("./data/eng/fake_news.json") as file:
        fake_news = json.load(file)


    fake_news_df = json2df(fake_news)

    compact_df = pd.concat([real_news_df, fake_news_df], ignore_index=True)

    # train_df, test_df = train_test_split(compact_df, test_size=0.2, random_state=2)

    k_CV(compact_df, model_dir='./model/model_src_kv-ENG')
