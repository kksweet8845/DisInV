import json
from Classifier.TriFn import json2df, k_CV
from sklearn.model_selection import train_test_split
import pandas as pd



if __name__ == "__main__":


    with open("./data/real_news_token_data.json") as file:
        real_news = json.load(file)

    real_news_df = json2df(real_news)
    real_news_df['label'] = [1] * len(real_news_df)

    with open("./data/fake_news_token_data.json") as file:
        fake_news = json.load(file)


    fake_news_df = json2df(fake_news)
    fake_news_df['label'] = [0] * len(fake_news_df)

    compact_df = pd.concat([real_news_df, fake_news_df], ignore_index=True)

    train_df, test_df = train_test_split(compact_df, test_size=0.2, random_state=2)

    k_CV(compact_df, model_dir="./model/model_src_kv")
