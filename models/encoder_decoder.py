import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

import warnings

warnings.filterwarnings('ignore')


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleEncoder(nn.Module):
    def __init__(self, input_channels, embedding_dim, is_embedding=False):
        super(SimpleEncoder, self).__init__()
        self.is_embedding = is_embedding
        if is_embedding:
            self.embedding = nn.Embedding(num_embeddings=700, embedding_dim=embedding_dim)
        self.fc1 = nn.Linear(in_features=input_channels, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        if self.is_embedding:
            embedding = self.embedding(x)
            return embedding, embedding
        x = self.relu(self.fc2(self.relu(self.fc1(x)))).squeeze(1)
        return x, x


class SimpleDecoder(nn.Module):
    def __init__(self, output_channels):
        super(SimpleDecoder, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc2(self.relu(self.fc1(x))))
        return x


class NeuralClusteringModel(nn.Module):
    def __init__(self, class_num, dim, df):
        super().__init__()
        self.input_dim = dim
        self.class_num = class_num
        self.df = df
        self.encoder = SimpleEncoder(input_channels=dim, embedding_dim=dim).to(device)
        self.decoder = SimpleDecoder(output_channels=dim).to(device)
        self.cluster_layer = nn.Linear(10, class_num, bias=False)  # clustering the decoder layer which is of length 10
        # make cluster random class_num points
        encoded_layer, _ = self.encode(torch.tensor(self.df.values).float().to(device))
        indices = np.random.choice(encoded_layer.shape[0], self.class_num, replace=False)
        random_points = encoded_layer[indices, :]
        self.cluster_layer.weight.data = random_points

        self.cluster_center = torch.rand([class_num, 10], requires_grad=False).to(device)
        self.inertia = self.overall_inertia()

    def overall_inertia(self):
        data = torch.tensor(self.df.values).to(torch.float)
        data = F.normalize(data, dim=1) * 2.0 * 0.001
        data = data.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=self.class_num, n_init=self.class_num)
        kmeans.fit(data)
        return kmeans.inertia_

    def encode(self, x):
        x, embedding = self.encoder(x)
        x = F.normalize(x)
        return x, embedding

    def decode(self, x):
        return self.decoder(x)

    def cluster(self, z):
        return self.cluster_layer(z)

    def normalize_cluster_center(self, alpha):
        self.cluster_layer.weight.data = (F.normalize(self.cluster_layer.weight.data, dim=1) * 2.0 * alpha)

    def compute_cluster_center(self, alpha):
        self.cluster_center = 1.0 / (2 * alpha) * self.cluster_layer.weight
        return self.cluster_center

    def visualize_cluster_centers(self, alpha, label=""):
        centers = self.compute_cluster_center(alpha)
        # pca reduction
        reducer = PCA(n_components=2)
        reduced_centers = reducer.fit_transform(centers.cpu().detach().numpy())
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title(f'Centers Visualization\n{label}', fontsize=20)
        ax.scatter(reduced_centers[:, 0], reduced_centers[:, 1], color='hotpink')
        ax.grid()
        plt.savefig(label.split("W")[0]) if label != "" else None
        # plt.show()
        plt.clf()

    def compute_pca_df(self, data, with_label=False):
        # get clustered data
        z, _, hard_label = self.cluster_data(data)
        # pca reduction
        pca = PCA(n_components=2)
        reduced_data_encoding = pca.fit_transform(z.cpu().detach().numpy())
        pca_cluster_df = pd.DataFrame(data=reduced_data_encoding, columns=['pca 1', 'pca 2'])
        pca_cluster_df['cluster'] = hard_label.cpu().detach().numpy().tolist()
        if with_label:
            data_labels = torch.cat(data[2])
            pca_cluster_df["label"] = data_labels.cpu().detach().numpy().astype(int)
        return pca_cluster_df

    def visualize_clustered_data(self, data, mi=None, with_label=False, label=""):
        pca_cluster_df = self.compute_pca_df(data, with_label=with_label)
        colors = plt.cm.rainbow(np.linspace(0, 1, self.class_num))
        # colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
        markers = ["o", "x"]  # square for 0 and x for 1

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        clusters = list(range(10))
        if with_label:
            ax.set_title(f'Labels (markers) and clusters (colors) after reduction, MI is: {mi}\n{label}', fontsize=20)
            for index in pca_cluster_df.index:
                color = colors[pca_cluster_df.loc[index, "cluster"]]
                marker = markers[pca_cluster_df.loc[index, "label"]]
                ax.scatter(pca_cluster_df.loc[index, "pca 1"], pca_cluster_df.loc[index, "pca 2"], c=color, s=50,
                           marker=marker)
            plt.savefig("label and markers " + label.split("W")[0])
        else:  # this was good for plotting clusters without labels
            ax.set_title('Clusters (colors) after reduction ', fontsize=20)
            for target, color in zip(clusters, colors):
                indicesToKeep = pca_cluster_df['cluster'] == target
                ax.scatter(pca_cluster_df.loc[indicesToKeep, 'pca 1']
                           , pca_cluster_df.loc[indicesToKeep, 'pca 2']
                           , c=color
                           , s=50)
            plt.savefig("clusters" + label.split("W")[0])
        # plt.show()
        plt.clf()

    def cluster_data(self, data):
        # process data to correct format
        data_x_tensor = torch.cat(data[0])
        data_x_tensor = data_x_tensor.to(device).float()

        # encode
        z, embedding = self.encode(data_x_tensor)

        # cluster
        clustered_data = self.cluster(z)

        # soft label
        soft_label = F.softmax(clustered_data.detach(), dim=1)
        # custer the data to the appropriate "hard label"
        hard_label = torch.argmax(soft_label, dim=1)

        return z, soft_label, hard_label

    def compute_labels_cluster_truth_data(self, data):
        pca_cluster_df = self.compute_pca_df(data, with_label=True)

        # count how many of each label in a cluster
        clusters = pca_cluster_df["cluster"].unique().tolist()
        sum_dict = {}
        for cluster in clusters:
            cluster_0_len = len(pca_cluster_df[(pca_cluster_df["cluster"] == cluster) & (pca_cluster_df["label"] == 0)])
            cluster_1_len = len(pca_cluster_df[(pca_cluster_df["cluster"] == cluster) & (pca_cluster_df["label"] == 1)])
            sum_dict[cluster] = {0: cluster_0_len, 1: cluster_1_len}
        cluster_count_df = pd.DataFrame(sum_dict).T
        print(f"cluster count dict is: {cluster_count_df}")
        print(f"mutual information is: ",
              normalized_mutual_info_score(pca_cluster_df["cluster"], pca_cluster_df["label"]))
        return normalized_mutual_info_score(pca_cluster_df["cluster"], pca_cluster_df["label"])

    def empty_cluster_handler(self, data, alpha):
        cnt = 0
        while True:
            # encode data to clusters
            data_x = torch.cat(data[0])
            z, soft_label, hard_label = self.cluster_data(data)

            # find all unique
            full_clusters, counts = np.unique(hard_label.cpu().detach().numpy(), return_counts=True)
            cluster_options = list(range(self.class_num))
            empty_clusters = list(set(cluster_options).difference(full_clusters.tolist()))
            # print(f"empty clusters are: {empty_clusters}")
            if not empty_clusters:
                break
            sse = nn.MSELoss(reduction="none")
            cluster_centers = self.compute_cluster_center(alpha)
            sse_values = torch.zeros(z.shape[0]).to(device)
            for cluster in full_clusters.tolist():  # only on clusters that have points assigned to
                cluster_center = cluster_centers[cluster]  # calculate center
                locations = (hard_label == cluster).nonzero(as_tuple=True)[0]  # get cluster data points
                data_points_in_cluster = z[locations]
                target_center = torch.stack([cluster_center] * len(data_points_in_cluster))  # reshape target sse
                sse_to_all_points_in_cluster = torch.sum(sse(data_points_in_cluster, target_center), dim=1)
                sse_values = sse_values.scatter(0, locations, sse_to_all_points_in_cluster)  # insert
            # non-iterative approach
            top_indexes = torch.topk(sse_values, len(empty_clusters)).indices
            for i, index in enumerate(top_indexes.tolist()):
                new_cluster = z[index]  # get the new center
                self.cluster_layer.weight.data[empty_clusters[i]] = new_cluster  # swap the empty cluster out
                # break #dev
            cnt += 1
            print(f"empty clusters: ")
            if cnt >= 10:
                break

    def split_clusters(self, empty_cluster, cluster_to_split, z, hard_label):
        # find all data points of that cluster
        cluster_indexes = np.where(hard_label.cpu().detach().numpy() == cluster_to_split)
        cluster_data = z[cluster_indexes]

        # cluster data into two new clusters
        kmeans = KMeans(n_clusters=2).fit(cluster_data.cpu().detach().numpy())
        new_centers = kmeans.cluster_centers_

        # swap the rows of the cluster layer to split rows
        self.cluster_layer.weight.data[empty_cluster] = torch.from_numpy(new_centers[0])
        self.cluster_layer.weight.data[cluster_to_split] = torch.from_numpy(new_centers[1])
