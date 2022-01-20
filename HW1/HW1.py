import numpy as np
import pandas as pd
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def remove_non_alphabetic(text):
    pattern = re.compile('[^a-zA-Z]')
    return pattern.sub(' ', text)


def remove_url(text):
    pattern = '[http|https]*://[a-zA-Z0-9.?#/&-_=:]*'
    url_pattern = re.compile(pattern)
    return url_pattern.sub('', text)


def remove_html(text):
    return BeautifulSoup(text).get_text()


def remove_space(text):
    space_pattern = re.compile('\s+')
    text = re.sub(space_pattern, ' ', text)
    return text


def to_lowercase(text):
    return text.lower()


def expand_contractions(text):
    return contractions.fix(text)


def report_length(step, df_sp):
    len_review = 0
    for review in df_sp["review"]:
        len_review += len(review)
    avglen = len_review / df_sp.shape[0]

    print("Average length " + step + ": " + str(avglen))


if __name__ == '__main__':
    # data preparation
    data_path = "./data/amazon_reviews_us_Kitchen_v1_00.tsv"
    df_ori = pd.read_csv(data_path, sep='\t', usecols=[7, 12, 13])

    label = []

    for star in df_ori["star_rating"]:
        if star > 3.0:
            label.append(1)
        elif star <= 2.0:
            label.append(0)
        else:
            label.append(-1)

    df_ori["label"] = label

    df_dnan = df_ori.dropna(axis=0, how="any")  # remove nan
    df_neu = df_dnan[df_dnan["label"] == -1]  # neutral review

    df_d3 = df_dnan.drop(df_dnan[(df_dnan["label"] == -1)].index)  # remove neutral review

    df_pos = df_d3[df_d3["label"] == 1]  # positive review
    df_neg = df_d3[df_d3["label"] == 0]  # negative review

    # report
    print("Number of positive sentiment: " + str(df_pos.shape[0]))
    print("Number of negative sentiment: " + str(df_neg.shape[0]))
    print("Number of neutral reviews: " + str(df_neu.shape[0]))

    # sampling
    df_pos_sp = df_pos.sample(n=100000)
    df_neg_sp = df_neg.sample(n=100000)
    df_sp = pd.concat([df_pos_sp, df_neg_sp])

    # data preprocessing
    df_sp["review"] = df_sp["review_headline"] + " " + df_sp["review_body"]
    df_sp.drop(columns=["star_rating", "review_headline", "review_body"])
    col_order = ["review", "label"]
    df_sp = df_sp[col_order]

    report_length("before cleaning", df_sp)  # average length before cleaning

    df_sp["review"] = df_sp["review"].apply(lambda x: to_lowercase(x))
    df_sp["review"] = df_sp["review"].apply(lambda x: remove_url(x))
    df_sp["review"] = df_sp["review"].apply(lambda x: remove_html(x))
    df_sp["review"] = df_sp["review"].apply(lambda x: expand_contractions(x))
    df_sp["review"] = df_sp["review"].apply(lambda x: remove_non_alphabetic(x))
    df_sp["review"] = df_sp["review"].apply(lambda x: remove_space(x))

    report_length("after cleaning", df_sp)  # average length after cleaning / before preprocessing

    # data preprocessing
    df_cd = df_sp

    stopwords = set(stopwords.words('english'))
    lem_words = []
    wnl = WordNetLemmatizer()
    report_length("before data preprocessing", df_sp)
    df_cd["review"] = df_cd["review"].apply(lambda x: word_tokenize(x))
    df_cd["review"] = df_cd["review"].apply(lambda x: " ".join([word for word in x if word not in stopwords]))

    df_cd["review"] = df_cd["review"].apply(lambda x: word_tokenize(x))
    df_cd["review"] = df_cd["review"].apply(lambda x: " ".join([wnl.lemmatize(word) for word in x]))

    report_length("after data preprocessing", df_cd)  # average length after data preprocessing

    # Feature Extraction
    reviews = []
    labels = []

    for i in df_cd["review"]:
        reviews.append(i)

    for i in df_cd["label"]:
        labels.append(i)

    review_array = np.array(reviews)
    label_array = np.array(labels)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(review_array)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    print("Perceptron")
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    print("Train")
    print("Accuracy: " + str(accuracy_score(pred_train, y_train)))
    print("Precision: " + str(precision_score(pred_train, y_train)))
    print("Recall: " + str(recall_score(pred_train, y_train)))
    print("F1: " + str(f1_score(pred_train, y_train)))

    pred_test = clf.predict(X_test)
    print("Test")
    print("Accuracy: " + str(accuracy_score(pred_test, y_test)))
    print("Precision: " + str(precision_score(pred_test, y_test)))
    print("Recall: " + str(recall_score(pred_test, y_test)))
    print("F1: " + str(f1_score(pred_test, y_test)))

    print("SVM")
    svm = LinearSVC(max_iter=1000)
    svm.fit(X_train, y_train)
    pred_train = svm.predict(X_train)
    print("Train")
    print("Accuracy: " + str(accuracy_score(pred_train, y_train)))
    print("Precision: " + str(precision_score(pred_train, y_train)))
    print("Recall: " + str(recall_score(pred_train, y_train)))
    print("F1: " + str(f1_score(pred_train, y_train)))

    pred_test = svm.predict(X_test)
    print("Test")
    print("Accuracy: " + str(accuracy_score(pred_test, y_test)))
    print("Precision: " + str(precision_score(pred_test, y_test)))
    print("Recall: " + str(recall_score(pred_test, y_test)))
    print("F1: " + str(f1_score(pred_test, y_test)))

    print("Logistic Regression")
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    pred_train = lr.predict(X_train)
    print("Train")
    print("Accuracy: " + str(accuracy_score(pred_train, y_train)))
    print("Precision: " + str(precision_score(pred_train, y_train)))
    print("Recall: " + str(recall_score(pred_train, y_train)))
    print("F1: " + str(f1_score(pred_train, y_train)))

    pred_test = lr.predict(X_test)
    print("Test")
    print("Accuracy: " + str(accuracy_score(pred_test, y_test)))
    print("Precision: " + str(precision_score(pred_test, y_test)))
    print("Recall: " + str(recall_score(pred_test, y_test)))
    print("F1: " + str(f1_score(pred_test, y_test)))

    print("Multinomial Naive Bayes")
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    pred_train = mnb.predict(X_train)
    print("Train")
    print("Accuracy: " + str(accuracy_score(pred_train, y_train)))
    print("Precision: " + str(precision_score(pred_train, y_train)))
    print("Recall: " + str(recall_score(pred_train, y_train)))
    print("F1: " + str(f1_score(pred_train, y_train)))

    pred_test = mnb.predict(X_test)
    print("Test")
    print("Accuracy: " + str(accuracy_score(pred_test, y_test)))
    print("Precision: " + str(precision_score(pred_test, y_test)))
    print("Recall: " + str(recall_score(pred_test, y_test)))
    print("F1: " + str(f1_score(pred_test, y_test)))
