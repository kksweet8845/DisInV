import pandas as pd
import json



if __name__ == "__main__":

    # open real dataset

    real1_df = pd.read_csv("./data/gossipcop_real.csv")
    real2_df = pd.read_csv("./data/politifact_real.csv")


    real_df = pd.concat([real1_df, real2_df], ignore_index=True)

    # open fake dataset

    fake1_df = pd.read_csv("./data/gossipcop_fake.csv")
    fake2_df = pd.read_csv("./data/politifact_fake.csv")

    fake_df = pd.concat([fake1_df, fake2_df], ignore_index=True)



    real_json_ls = []

    for row in real_df.iterrows():
        row = row[1]
        # print(row)
        tmp = {
            'title' : row['title'],
            'brand_id' : 0,
            'author' : "FakeNewsNet",
            'url' : row['news_url'],
            'label' : 1,
            'standpoint' : 0,
            'token_title' : list(row['title'].split(' ')),
            "content" : "",
            "date" : "2021-01-01"
        }
        real_json_ls.append(tmp)

    json_str = json.dumps(real_json_ls, indent=4, ensure_ascii=False)


    with open("./data/eng/real_news.json", "w", encoding="utf8") as file:
        file.write(json_str)
    file.close()


    fake_json_ls = []

    for row in fake_df.iterrows():
        row = row[1]
        tmp = {
            'title' : row['title'],
            'brand_id' : 0,
            'author' : 'FakeNewsNet',
            'url' : row['news_url'],
            'label' : 0,
            'standpoint' : 0,
            'token_title' : list(row['title'].split(' ')),
            'content' : "",
            "date" : "2021-01-01"
        }
        fake_json_ls.append(tmp)

    json_str = json.dumps(fake_json_ls, indent=4, ensure_ascii=False)

    with open("./data/eng/fake_news.json", "w", encoding="utf8") as file:
        file.write(json_str)
    file.close()

