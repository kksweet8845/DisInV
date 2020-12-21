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



if __name__ == '__main__':

    with open('./data/brand_id_4.json') as file:
        _4_news = json.load(file)

    left_news_df = json2df(_4_news)
    left_news_df['label'] = [0]*len(left_news_df)

    with open('./data/brand_id_8.json') as file:
        _8_news = json.load(file)

    right_news_df = json2df(_8_news)
    right_news_df['label'] = [1]*len(right_news_df)


    compact_df = pd.concat([left_news_df, right_news_df[:len(left_news_df)]], ignore_index=True)

    X = compact_df['title']
    y = compact_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

    train_data = list(zip(X_train, y_train))
    train_df = pd.DataFrame(train_data, columns=['text', 'labels'])

    test_data = list(zip(X_test, y_test))
    test_df = pd.DataFrame(test_data, columns=['text', 'labels'])

    model_args = ClassificationArgs(num_train_epochs=10)
    model = ClassificationModel("bert", "bert-base-chinese", args=model_args, use_cuda=True)

    model.train_model(train_df)


    result, model_outputs, wrong_predictions = model.eval_model(test_df)

    with open("./bert.log", "w+") as file:
        print(result, file=file)
        print(model_outputs, file=file)
        print(wrong_predictions, file=file)






