from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

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
            row['url']
        ])

    col = [
            'title',
            'content',
            'author',
            'brand_id',
            'date',
            'url'
        ]

    return pd.DataFrame(ls, columns=col)


def df2json(df, filename):

    dump_json = []
    for row in df.iterrows():
        row = row[1]
        obj = {
            'title' : row['title'],
            'content' : row['content'],
            'author' : row['author'],
            'brand_id' : row['brand_id'],
            'date' : str(row['date']),
            'url' : row['url'],
            'standpoint' : row['standpoint']
        }
        dump_json.append(obj)

    json_str = json.dumps(dump_json, indent=4, ensure_ascii=False)

    with open(filename, "w+", encoding='utf8') as file:
        file.write(json_str)

    file.close()





if __name__ == '__main__':

    with open('./data/real_news_dump.json') as file:
        real_news = json.load(file)

    real_news_df = json2df(real_news)

    with open('./data/fake_news_dump.json') as file:
        fake_news = json.load(file)

    fake_news_df = json2df(fake_news)

    compact_df = pd.concat([real_news_df, fake_news_df], ignore_index=True)

    X = compact_df['title']

    model_args = ClassificationArgs()
    model = ClassificationModel("bert", "./outputs", args=model_args, use_cuda=True)

    # predictions, raw_outputs = model.predict(X)

    X_real = real_news_df['title']

    real_pred, raw_outputs = model.predict(X_real)
    real_news_df['standpoint'] = real_pred


    df2json(real_news_df, "./data/real_news_data.json")

    X_fake = fake_news_df['title']

    fake_pred, raw_outputs = model.predict(X_fake)
    fake_news_df['standpoint'] = fake_pred


    df2json(fake_news_df, "./data/fake_news_data.json")



