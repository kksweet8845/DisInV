from sklearn.model_selection import train_test_split
import json
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_random_state
from Classifier.TriFn import json2df
from sklearn import preprocessing, model_selection, feature_extraction, naive_bayes, metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def k_CV(clf, x, Y, clf_name):

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    _clf = Pipeline([
        ('tf-idf', feature_extraction.text.TfidfVectorizer()),
        ('clf', clf)
    ])

    scores = cross_validate(_clf, x, Y, scoring=scoring)


    print("=" * 60)
    print("\n")
    print(f"Classifier name    : {clf_name}")
    print(f"Accuracy           : {scores['test_accuracy']}")
    print(f"Precision          : {scores['test_precision_macro']}")
    print(f"Recall             : {scores['test_recall_macro']}")
    print(f"F1                 : {scores['test_f1_macro']}")
    print(f"Accuracy mean      : {np.mean(scores['test_accuracy'])}")
    print(f"Precision mean     : {np.mean(scores['test_precision_macro'])}")
    print(f"Recall mean        : {np.mean(scores['test_recall_macro'])}")
    print(f"F1 mean            : {np.mean(scores['test_f1_macro'])}")
    print("\n")
    print("=" * 60)


if __name__ == "__main__":

    with open("./data/eng/real_news.json") as file:
        real_news = json.load(file)

    real_news_df = json2df(real_news)

    with open("./data/eng/fake_news.json") as file:
        fake_news = json.load(file)

    fake_news_df = json2df(fake_news)

    real_news_df['label'] = [1] * len(real_news_df)
    fake_news_df['label'] = [0] * len(fake_news_df)

    fake_news_df = fake_news_df.dropna()
    real_news_df = real_news_df.dropna()

    compact_df = pd.concat([real_news_df, fake_news_df], ignore_index=True)
    X = compact_df['title']
    y = compact_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

    

    # Naive Bayes BernoulliNB()
    clf = naive_bayes.BernoulliNB()
    k_CV(clf, X_train, y_train, "BernoulliNB()")

    # SGDClassifier()
    k_CV(SGDClassifier(), X_train, y_train, "SGDClassifier()")

    # AdaBoostClassifier
    k_CV(AdaBoostClassifier(), X_train, y_train, "AdaBoostClassifier()")

    # DecisionTreeClassifier
    k_CV(DecisionTreeClassifier(), X_train, y_train, "DecisionTreeClassifier()")

    # RandomForestClassifier
    k_CV(RandomForestClassifier(), X_train, y_train, "RandomForestClassifier()")

    # ComplementNB
    k_CV(naive_bayes.ComplementNB(), X_train, y_train, "ComplementNB()")

