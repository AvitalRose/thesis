import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss is a loss function that is used to learn cross-modal embeddings it focuses on comparing the
    similarity or dissimilarity of vectors.
    The goal of contrastive loss is to bring similar instances closer together in the embedding space and push apart
    dissimilar instances.
    """
    def __init__(self, batch_size, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        :params emb_i: Made from splitting see above
        :params emb_j: Made from splitting see above
        :return:
        """
        batch_size = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
        # denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss


class TriangleLoss(nn.Module):
    """
    Triangle Loss class to calculate the triangle loss. The loss is calculated as an approximation of the real loss
    since it is computationally expensive otherwise.
    """
    def __init__(self, batch_size, temperature=0.5):
        super(TriangleLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, embedding):
        """
        :param embedding:
        """
        n = embedding.shape[0]
        representations = F.normalize(embedding, dim=1)  # why normalize?
        # find similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        exp_similarity = torch.exp(similarity_matrix / self.temperature)
        exp_similarity = F.normalize(exp_similarity, dim=1)
        double_edge_matrix = torch.matmul(exp_similarity, exp_similarity)
        double_edge_matrix = torch.div(double_edge_matrix, n * n)
        # subtracted_matrix = torch.sub(exp_similarity, double_edge_matrix, alpha=1/exp_similarity.shape[0])
        subtracted_matrix = torch.sub(double_edge_matrix, exp_similarity, alpha=n)
        sum_matrix = torch.sum(subtracted_matrix) / (n * n)
        return - sum_matrix  # so we are minimizing


class CustomDistanceLoss(nn.Module):
    def __init__(self, batch_size, weights=None, params=None):
        """
        The custom loss class, includes all the different components of the loss:
        1. separation loss
        2. invariance loss
        3. distinction loss
        4. triangular loss
        :param weights: how much weight to give each loss
        """
        super(CustomDistanceLoss, self).__init__()
        self.batch_size = batch_size
        self.weights = weights if weights else {"sep": 1, "inv": 0, "dist": 0, "tri": 0}
        self.params = params if params else {"temp": 0.05, "k": 10}
        self.instances_num = 2
        self.deep_to_shallow_ratio = 2  # how much of the bag embedding to cut out for the invariance loss

        # loss components
        self.contrastive_loss = ContrastiveLoss(batch_size=batch_size)
        self.tri_loss = TriangleLoss(batch_size=batch_size)
        self.loss_dict = {"sep": 0, "inv": 0, "dist": 0, "tri": 0}

    def forward(self, outputs, bag_labels):
        # the forward pass of the contrastive loss takes two matrices where the corresponding indices are positives
        # so must slice encoded into two
        self.loss_dict["sep"] = self.compute_separation_loss(outputs, bag_labels)
        self.loss_dict["inv"] = self.compute_invariance_loss(outputs, bag_labels)
        self.loss_dict["tri"] = self.tri_loss(embedding=outputs)
        return self.loss_dict["sep"], self.loss_dict["tri"], self.loss_dict["inv"]

    def compute_separation_loss(self, embedding, labels):
        # select two from each length
        unique_labels = torch.unique(labels, sorted=False)
        double_embedding_list = []
        for bag_label in unique_labels:
            # get the rows of the encoded abg
            bag_embedding = embedding[(labels == bag_label).nonzero().squeeze(1)]
            if bag_embedding.shape[0] == 1:
                continue  # can't perform separation loss here, will just not compute for that bag
            elif bag_embedding.shape[0] == self.instances_num:
                double_embedding_list.append(bag_embedding)
            else:
                indices = np.random.choice(bag_embedding.shape[0], self.instances_num, replace=False)
                double_embedding_list.append(bag_embedding[indices])

        # this part is for dev:
        if len(double_embedding_list) == 0:
            print(f"No instances that are doubled, size of all embedding is: {embedding.shape}")
            return torch.tensor(1000)

        double_embedding = torch.cat(double_embedding_list)
        double_embedding = double_embedding.reshape((int(double_embedding.shape[0] / self.instances_num),
                                            self.instances_num, double_embedding.shape[-1]))

        emb_i = double_embedding[:, 0, :]
        emb_j = double_embedding[:, 1, :]
        separation_loss = self.contrastive_loss(emb_i, emb_j)
        return separation_loss

    def compute_invariance_loss(self, embedding, labels):
        unique_labels = torch.unique(labels, sorted=False)
        deep_bag_embedding_list = []
        shallow_bag_embedding_list = []
        for bag_label in unique_labels:
            # get the rows of the encoded abg
            bag_embedding = embedding[(labels == bag_label).nonzero().squeeze(1)]
            # get the full bag
            sum_bag_embeddings_vector = torch.sum(bag_embedding, dim=0).unsqueeze(0)
            deep_bag_embedding_list.append(sum_bag_embeddings_vector)
            # get the "shallow" bag
            num_of_indices = int(bag_embedding.shape[0] / self.deep_to_shallow_ratio)
            indices = np.random.choice(bag_embedding.shape[0], num_of_indices, replace=False)
            sum_shallow_bag_embeddings_vector = torch.sum(bag_embedding[indices], dim=0).unsqueeze(0)
            shallow_bag_embedding_list.append(sum_shallow_bag_embeddings_vector)
        deep_bag_embedding = torch.cat(deep_bag_embedding_list)
        shallow_bag_embedding = torch.cat(shallow_bag_embedding_list)
        invariance_loss = self.contrastive_loss(deep_bag_embedding, shallow_bag_embedding)
        return invariance_loss

