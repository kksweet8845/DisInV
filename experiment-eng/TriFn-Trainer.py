from Classifier.TriFn import TriFn_Trainer, json2df, k_CV, TriFn_dataset
import json
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

    train_df, test_df = train_test_split(compact_df, test_size=0.2, random_state=2)

    train_data = TriFn_dataset(train_df)
    test_data = TriFn_dataset(test_df)

    train = True

    if train == True:
        trainer = TriFn_Trainer('cuda', train_data, './model_src-ENG', pretrained=False, test=test_data, batch_size=64)
        trainer.train(10, 'TriFn_eng_v1')
    elif train == False:
        pass
