import json



if __name__ == "__main__":

    with open("./data/fake_news_dump.json") as file:
        fake_js = json.load(file)

    with open("./data/real_news_2021-02-20_dump.json") as file:
        real_js = json.load(file)

    
    print(f"Real news num: {len(real_js)}")
    print(f"Fake news num: {len(fake_js)}")