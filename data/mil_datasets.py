import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pandas as pd


class MILDataset(Dataset):
    """
    The dataset can be given as the data itself or as an embedding keys of the data, depending on the embedding flag.
    If True:
    The data will be processed into a dictionary of key:index for each data point. The instance returned will be of the
    index of each of the data keys.
    If false:
    The direct data is returned.
    In previous version, the dataset can be organized in two ways, according to how the Bags flag is set.
    If True:
    Each instance of the data is a bag of single example instances.
    Else:
    Each instance of the data is a single example instance.
    In previous version there was a pairs flag which indicates if the whole bag should be returned or two instances
    at a time.
    In previous version there was a instances_num variable which indicates the number of instances to return from a bag.
    """

    def __init__(self, df, embedding=False, multi_label=False):
        self.df = df
        self.X = []
        self.y = []
        self.label = []
        # preprocess data
        self.preprocess_data()
        self.molc = list(set(list(self.df.index)))
        self.molc_dict = {m: i for i, m in enumerate(self.molc)}
        self.molc_dict_idx_to_label = {i: m for m, i in self.molc_dict.items()}
        self.embedding = embedding
        self.multi_label = multi_label

        if self.embedding:
            self.word_to_ix = self.get_embedding()

        # create the bags
        self.bags_aggregate()

        # drop the label colum
        self.df = self.df.drop("label", axis=1)
        self.data_df = self.df
        self.data_df.columns = self.data_df.columns.astype(str)

    def preprocess_data(self):
        scaler = MinMaxScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), index=self.df.index)
        # adding label colum after the fit transform
        self.df["label"] = self.df.index.str.contains("NON").astype(int)

    def get_embedding(self):
        # find all possible "words"
        l1 = []
        for col in self.df.columns:
            l1.extend(self.df[col].unique())
        l1_set = set(l1)

        # convert to dictionary
        word_to_ix = {word: i for i, word in enumerate(l1_set)}
        return word_to_ix

    def bags_aggregate(self):
        for m in self.molc:
            bag = self.df[self.df.index == m]
            label = bag['label'].values
            self.X.append(torch.tensor(bag.loc[:, self.df.columns != "label"].values))
            self.label.append(torch.tensor(label))
            y_label = self.molc_dict[m]
            self.y.append(torch.tensor([y_label]*bag.values.shape[0]))

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter < len(self.y):
            to_return = [self.X[self.iter], self.y[self.iter]]
            self.iter += 1
            return to_return
        else:
            raise StopIteration

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        if self.embedding:
            x = self.X[idx]
            x_ix = []
            for row in x:  # turn row into tensor of words
                rows_ix = [self.word_to_ix[value.item()] for value in row]
                x_ix.append(torch.unsqueeze(torch.as_tensor(rows_ix), 0))
            x = torch.cat(x_ix, dim=0)
            y = self.y[idx]
            label = self.label[idx]
        else:
            label = self.label[idx]
            x = self.X[idx]
            y = self.y[idx]
        return x, y, label

    @property
    def data(self):
        if self.embedding:
            x = self.X
            # turn row into tensor of words
            x_ix = []
            for bag in x:
                bag_tensor = []
                for row in bag:
                    rows_ix = [self.word_to_ix[value.item()] for value in row]
                    bag_tensor.append(torch.unsqueeze(torch.as_tensor(rows_ix), 0))
                bag = torch.cat(bag_tensor, dim=0)
                x_ix.append(bag)
            x = x_ix
            y = self.y
            label = self.label
            return x, y, label
        return self.X, self.y, self.label


class MuskDataset(MILDataset):
    """
    Musk Datasets
    -------------
    musk1:
    92 Bags
    47 positive
    45 negative
    """
    def __init__(self, df, embedding=False):
        super().__init__(df, embedding=embedding)

    def preprocess_data(self):
        self.df = self.df.drop([1, 168], axis=1)  # this is for musk1.data not needed for original nodes
        v_threshold = VarianceThreshold(threshold=0)
        self.df = pd.DataFrame(v_threshold.fit_transform(self.df), index=self.df.index)
        scaler = MinMaxScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), index=self.df.index)

        # adding label colum after the fit transform
        self.df["label"] = self.df.index.str.contains("NON").astype(int)


class AnimalDataset(MILDataset):
    """
    This is the dataset class for the image retrieval data, which contains three datasets - FOX, TIGER & ELEPHANT
    According to https://arxiv.org/pdf/1806.08186v1.pdf
    Image Retrival Datasets
    -------------
    FOX:
    200 Bags
    100 positive
    100 negative
    TIGER:
    200 Bags
    100 positive
    100 negative
    ELEPHANT:
    200 Bags
    100 positive
    100 negative
    """
    def __init__(self, df, embedding=False):
        super().__init__(df, embedding=embedding)

    def preprocess_data(self):
        self.df = self.df.iloc[:, 0].str.split(" ", expand=True)
        self.df[["num", "bag", "label"]] = self.df.iloc[:, 0].str.split(":", expand=True).astype(int)
        self.df = self.df.drop([0, 231], axis=1)
        for column in self.df.columns:
            if column not in ["num", "bag", "label"]:
                self.df[column] = self.df[column].str.split(":", expand=True)[1]
        self.df = self.df.set_index("bag")
        self.df = self.df.drop("num", axis=1)
        self.df = self.df.astype(float)
        scaler = MinMaxScaler()
        df_labels = self.df["label"]
        self.df = pd.DataFrame(scaler.fit_transform(self.df.loc[:, self.df.columns != "label"]), index=self.df.index)

        self.df["label"] = df_labels
        self.df["label"] = self.df["label"].astype(int)
        self.df["label"] = self.df["label"].replace(-1, 0)
        return self.df


if __name__ == "__main__":
    web_path = r"C:\Users\avita\Documents\skl\ThesisProject\data\WEB\web1.csv"
    # web_path.

