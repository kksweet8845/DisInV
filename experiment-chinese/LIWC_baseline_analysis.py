"""
    Author  : Nober Tai
    github  : kksweet8845
    Purpose :
        Check the baseline of fake news detection with LIWC representation
"""
import json
import pandas as pd
import numpy as np
from Classifier.TriFn import json2df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def k_CV(clf, x, Y, clf_name):

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    scores = cross_validate(clf, x, Y, scoring=scoring)

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


    # Import dataset
    with open("./data/fake_news_LIWC_data.json") as file:
       fake_news = json.load(file)

    fake_df = json2df(fake_news)

    fake_df['label'] = [0] * len(fake_df)

    with open("./data/real_news_LIWC_data.json") as file:
        real_news = json.load(file)

    real_df = json2df(real_news)

    real_df['label'] = [1] * len(real_df)


    compact_df = pd.concat([fake_df, real_df], ignore_index=True)


    train_df, test_df = train_test_split(compact_df, test_size=0.1, random_state=2)

    x_train, Y_train = np.array(train_df['LIWC'].values.tolist(), dtype=np.float), np.array(train_df['label'].values.tolist())

    # x_test, Y_test = test_df['LIWC'], test_df['label']


    

    # Naive Bayes BernoulliNB()
    clf = naive_bayes.BernoulliNB()
    k_CV(clf, x_train, Y_train, "BernoulliNB()")

    # SBDClassifier
    k_CV(SGDClassifier(), x_train, Y_train, "SGDClassifier()")

    # AdaBoostClassifier
    k_CV(AdaBoostClassifier(), x_train, Y_train, "AdaBoostClassifier()")

    # DecisionTreeClassifier
    k_CV(DecisionTreeClassifier(), x_train, Y_train, "DecisionTreeClassifier()")

    # RandomForestClassifier
    k_CV(RandomForestClassifier(), x_train, Y_train, "RandomForestClassifier()")

    # ComplementNB
    k_CV(naive_bayes.ComplementNB(), x_train, Y_train, "ComplementNB()")
