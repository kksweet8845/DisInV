try:
    from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
except:
    pass

import json
import pandas as pd

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
            row['standpoint']
        ])

    col = [
            'title',
            'content',
            'author',
            'brand_id',
            'date',
            'url',
            'standpoint'
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
            'standpoint' : row['standpoint'],
            'token_title' : row['token_title']
        }
        dump_json.append(obj)

    json_str = json.dumps(dump_json, indent=4, ensure_ascii=False)

    with open(filename, "w+", encoding='utf8') as file:
        file.write(json_str)

    file.close()


class TriFn_tokenizer:
    def __init__(self):
        # data_utils.download_data_gdown("./ckiptagger")
        self.ws = WS("./ckiptagger/data")

    def tokenize(self, X):
        return self.ws(X)



if __name__ == '__main__':

    tokenizer = TriFn_tokenizer()

    with open('./data/real_news_data.json', 'r') as file:
        real_news = json.load(file)

    real_news_df = json2df(real_news)

    with open('./data/fake_news_data.json', 'r') as file:
        fake_news = json.load(file)

    fake_news_df = json2df(fake_news)

    real_news_df['token_title'] = tokenizer.tokenize(real_news_df['title'])
    
    fake_news_df['token_title'] = tokenizer.tokenize(fake_news_df['title'])

    df2json(real_news_df, './data/real_news_token_data.json')
    df2json(fake_news_df, './data/fake_news_token_data.json')

    