import pandas as pd
import json
import pickle as pkl
from collections import Counter
import matplotlib.pyplot as plt
import torch
from data.mil_datasets import MILDataset
from sklearn.feature_extraction.text import CountVectorizer
import re
import random
import nltk


# nltk.download('stopwords')

def clean_dataframe(data):
    # "drop nans, then apply 'clean_sentence' function to Description"
    data = data.dropna(how="any")
    for col in ['description']:
        data[col] = data[col].apply(clean_sentence)
    return data


def clean_sentence(val):
    # "remove chars that are not letters or numbers, downcase, then remove stop words"
    STOP_WORDS = nltk.corpus.stopwords.words()
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)
    sentence = " ".join(sentence)
    return sentence


class WikiDataset(MILDataset):
    """
    This is the dataset class for wiki dataset, which contains pages from the Wikipedia dataset scraped using the
    wiki scraper.
    Wikipedia Datasets
    -------------
    Each page is a bag, and each section in the page is an instance of the bag, in a bag of word format.
    Each bag labels can be a set of labels, which are the categories tags on the bottom of the page for multi label.
    or a single label when not multi-label.
    """
    def __init__(self, corp_path, df=None,  label_path=None, category_path=None, num_label=50, max_thresh=0.9,
                 min_thresh=0.01, embedding=False, multi_label=False, **kwargs):
        self.corp_path = corp_path
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.num_label = num_label
        self.multi_label = multi_label

        if isinstance(df, str):  # if string and not dataframe
            self.df, self.label2index = self.get_dataset()
            self.df.to_csv(df)
            with open(label_path, "wb") as f:
                pkl.dump(self.label2index, f)
            self.index2label = {idx: label for label, idx in self.label2index.items()}
            self.categories_counter = self.get_dataset_statistics()
            with open(category_path, "wb") as f:
                pkl.dump(self.categories_counter, f)
        else:
            self.df = df
            with open(label_path, "rb") as f:
                self.label2index = pkl.load(f)
            with open(category_path, "rb") as f:
                self.categories_counter = pkl.load(f)
            self.index2label = {idx: label for label, idx in self.label2index.items()}
        super().__init__(self.df, embedding=embedding, multi_label=multi_label)

    def train_test_split(self, ratio=0.8):
        df_with_labels = self.df.copy()
        df_with_labels["label"] = torch.cat(self.label)
        indexes = list(set(df_with_labels.index))
        random.shuffle(indexes)
        train_indexes = indexes[:int(len(indexes) * ratio)]
        val_indexes = indexes[int(len(indexes) * ratio):]
        train_df = df_with_labels.loc[train_indexes, :]
        val_df = df_with_labels.loc[val_indexes, :]
        return train_df, val_df

    def preprocess_data(self):
        # clean from labels to include only num_label amount
        # common_labels = [self.index2label[c[1]] for c in self.categories_counter.most_common(self.num_label)]
        print(f"self.categories_counter: {self.categories_counter}")
        common_labels = [int(c[0]) for c in self.categories_counter.most_common(self.num_label)]
        if self.multi_label:
            self.df["label"] = self.df["label"].apply(lambda x: list(set(x).intersection(common_labels)))
        return self.df

    def bags_aggregate(self):
        for m in self.molc:
            bag = self.df[self.df.index == m]
            label = bag['label']
            self.X.append(torch.tensor(bag.loc[:, self.df.columns != "label"].values))
            if self.multi_label:
                self.label.append(torch.tensor([label[0]]*bag.values.shape[0]))
            else:
                self.label.append(torch.tensor(label))
            y_label = self.molc_dict[m]
            self.y.append(torch.tensor([y_label]*bag.values.shape[0]))

    def get_dataset_statistics(self):
        # histogram of labels
        categories_counter = Counter()
        for bag, bag_group in self.df.groupby("bag"):
            if self.multi_label:
                categories_counter.update(bag_group["label"].iloc[0])
            else:
                categories_counter.update(bag_group["label"])
        sorted_categories_counter = dict(sorted(categories_counter.items(), key=lambda item: item[1], reverse=True))
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        d = {str(c): value for c, value in sorted_categories_counter.items()}
        plt.bar(d.keys(), d.values())
        plt.title(f"Label Histogram\n"
                  f"Number Of Unique Labels: {len(self.label2index)}")
        plt.ylabel('No. of Occurrences')
        plt.xticks(visible=False)
        # plt.show()

        # histogram of words
        df_only_words = self.df.drop(["label"], axis=1)
        sum_df = pd.DataFrame(df_only_words.sum()).sort_values(0, ascending=False)
        plt.bar(sum_df.index.tolist(), sum_df.loc[:, 0].astype(int).tolist(), color='blue')
        plt.xticks(visible=False)
        plt.ylabel('No. of Occurrences')
        plt.title(f'Word Histogram\nVocabulary Size: {len(sum_df.index)} min thresh: {self.min_thresh}'
                  f' and max thresh: {self.max_thresh}')
        # plt.show()
        return categories_counter

    def get_dataset(self):
        labels_dict, bag_dict = self.read_corpus()

        # get corpus df and concatenate with bag and label df
        bag_df = pd.DataFrame(bag_dict).T
        clean_df = clean_dataframe(bag_df)

        corpus_df: pd.DataFrame = self.vectorize_word_cont(clean_df["description"])
        bag_df = pd.concat([clean_df, corpus_df], axis=1)
        bag_df = bag_df.drop("description", axis=1)
        bag_df = bag_df.set_index("bag")
        return bag_df, labels_dict

    def vectorize_word_cont(self, corpus: list) -> pd.DataFrame:
        # instantiate the vectorizer
        vectorizer = CountVectorizer(max_df=self.max_thresh, min_df=self.min_thresh)

        # apply the vectorizer to the corpus
        X = vectorizer.fit_transform(corpus)

        # display the document-term matrix as a dataframe to show the tokens
        vocab = vectorizer.get_feature_names_out()
        docterm = pd.DataFrame(X.todense(), columns=vocab)

        return docterm

    def read_corpus(self):
        with open(self.corp_path, "r") as f:
            pages_dict = json.load(f)
        # corpus will only include words in sections, not section titles or  bag categories
        corpus_dict = {}
        categories_to_idx = {}
        cnt = 0
        for index, (page_name, pages_dict) in enumerate(pages_dict.items()):
            page_categories = []
            for category in pages_dict["categories"]:
                if category not in categories_to_idx:
                    categories_to_idx[category] = len(categories_to_idx) + 1
                if self.multi_label:
                    page_categories.append(categories_to_idx[category])
                else:
                    page_categories = categories_to_idx[category]
            for section_name, section_content in pages_dict["sections"].items():
                corpus_dict[cnt] = {"bag": page_name, "label": page_categories, "description": section_content}
                cnt += 1
        return categories_to_idx, corpus_dict

