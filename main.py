import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from utils import compute_kde, plot_data, save_data, print_losses, update_kde, update_params, change_params
from models.encoder_decoder import NeuralClusteringModel
from models.loss_model import CustomDistanceLoss
from data.mil_datasets import AnimalDataset, MuskDataset
from data.wiki_dataset import WikiDataset
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def my_collate(batch):
    """
    Can deal with both single and multi label options
    """
    data = torch.cat([item[0] for item in batch], dim=0)
    target = torch.cat([item[1] for item in batch], dim=0)
    if batch[0][2].dim == 1:
        label = torch.cat([item[2] for item in batch], dim=0)
    else:
        label = [item[2] for item in batch]
    return [data, target, label]


def train(model, data_loader, optimizer, criterion, custom_loss, epoch, data, loss_dict, params):
    """
    The end-to-end training framework. The reconstruction loss is used to optimize encoder and decoder simultaneously,
    while the clustering loss is used to optimize the cluster layer and encoder alternatively
    """
    weights = params["loss_weights"]
    loss_clu_epoch = loss_rec_epoch = loss_cont_epoch = loss_tri_epoch = loss_inv_epoch = overall_loss = 0
    model.train()
    for step, (x, y, label) in enumerate(data_loader):
        optimizer.zero_grad()
        x = x.to(device).float()
        z, embedding = model.encode(x)
        # cluster and calculate cluster loss and empty cluster handler
        if epoch % 2 == 1:
            cluster_batch = model.cluster(z)
        else:
            cluster_batch = model.cluster(z.detach())
        # split empty clusters
        empty_clusters = model.empty_cluster_handler(data, params["alpha"])
        if empty_clusters:
            # re cluster batch
            print(f"re cluster batch, empty clusters is: {empty_clusters}")
            cluster_batch = model.cluster(z.detach())
        soft_label = F.softmax(cluster_batch.detach(), dim=1)
        hard_label = torch.argmax(soft_label, dim=1)
        delta = torch.zeros((cluster_batch.shape[0], params["class_num"]), requires_grad=False).to(device)
        for i in range(cluster_batch.shape[0]):  # the batch size
            delta[i, torch.argmax(soft_label[i, :])] = 1  # to get distance only to chosen center (the max)
        loss_clu_batch = 2 * params["alpha"] - torch.mul(delta, cluster_batch)
        loss_clu_batch = 0.01 / params["alpha"] * loss_clu_batch.mean()

        # decode
        x_ = model.decode(z)
        # reconstruction decoder loss
        loss_rec = criterion(x, x_)
        mean_across_cols = torch.mean(x, -1).unsqueeze(-1).expand(-1, x.shape[-1])
        loss_from_mean = criterion(x, mean_across_cols)
        loss_rec /= loss_from_mean  # for normalizing

        # contrastive loss
        loss_cont, loss_tri, loss_inv = custom_loss(x_, y)

        # combine
        loss = (loss_rec * weights["reconstruction"] + loss_cont * weights["contrastive"] +
                loss_tri * weights["triangle"] + loss_inv * weights["invariance"] +
                loss_clu_batch * weights["clustering"])
        loss /= sum(weights.values())
        loss.backward()
        if epoch % 2 == 0:
            model.cluster_layer.weight.grad = (
                    F.normalize(model.cluster_layer.weight.grad, dim=1) * 0.2 * params["alpha"]
            )
        else:
            model.cluster_layer.zero_grad()
        optimizer.step()
        model.normalize_cluster_center(params["alpha"])

        # adding to loss list
        loss_clu_epoch += loss_clu_batch.item()
        loss_rec_epoch += loss_rec.item()
        loss_cont_epoch += loss_cont.item()
        loss_tri_epoch += loss_tri.item()
        loss_inv_epoch += loss_inv.item()
        overall_loss += loss.item()

    loss_dict["clustering"]["train"].append(loss_clu_epoch / len(data_loader))
    loss_dict["reconstruction"]["train"].append(loss_rec_epoch / len(data_loader))
    loss_dict["contrastive"]["train"].append(loss_cont_epoch / len(data_loader))
    loss_dict["triangle"]["train"].append(loss_tri_epoch / len(data_loader))
    loss_dict["invariance"]["train"].append(loss_inv_epoch / len(data_loader))
    loss_dict["all"]["train"].append(overall_loss / len(data_loader))


def validate(model, data_loader, criterion, custom_loss, epoch, data, loss_dict, params):
    weights = params["loss_weights"]
    loss_clu_epoch = loss_rec_epoch = loss_cont_epoch = loss_tri_epoch = loss_inv_epoch = overall_loss = 0
    model.eval()
    with torch.no_grad():
        for step, (x, y, label) in enumerate(data_loader):
            x = x.to(device).float()
            z, embedding = model.encode(x)
            # cluster and calculate cluster loss and empty cluster handler
            cluster_batch = model.cluster(z.detach())
            soft_label = F.softmax(cluster_batch.detach(), dim=1)
            hard_label = torch.argmax(soft_label, dim=1)
            delta = torch.zeros((cluster_batch.shape[0], params["class_num"]), requires_grad=False).to(device)
            for i in range(cluster_batch.shape[0]):  # the batch size
                delta[i, torch.argmax(soft_label[i, :])] = 1  # to get distance only to chosen center (the max)
            loss_clu_batch = 2 * params["alpha"] - torch.mul(delta, cluster_batch)
            loss_clu_batch = 0.01 / params["alpha"] * loss_clu_batch.mean()

            # decode
            x_ = model.decode(z)
            # reconstruction decoder loss
            loss_rec = criterion(x, x_)
            mean_across_cols = torch.mean(x, -1).unsqueeze(-1).expand(-1, x.shape[-1])
            loss_from_mean = criterion(x, mean_across_cols)
            loss_rec /= loss_from_mean  # for normalizing

            # contrastive loss
            loss_cont, loss_tri, loss_inv = custom_loss(x_, y)

            # combine
            loss = (loss_rec * weights["reconstruction"] + loss_cont * weights["contrastive"] +
                    loss_tri * weights["triangle"] + loss_inv * weights["invariance"] +
                    loss_clu_batch * weights["clustering"])
            loss /= sum(weights.values())

            # adding to loss list
            loss_clu_epoch += loss_clu_batch.item()
            loss_rec_epoch += loss_rec.item()
            loss_cont_epoch += loss_cont.item()
            loss_tri_epoch += loss_tri.item()
            loss_inv_epoch += loss_inv.item()
            overall_loss += loss.item()

        loss_dict["clustering"]["validation"].append(loss_clu_epoch / len(data_loader))
        loss_dict["reconstruction"]["validation"].append(loss_rec_epoch / len(data_loader))
        loss_dict["contrastive"]["validation"].append(loss_cont_epoch / len(data_loader))
        loss_dict["triangle"]["validation"].append(loss_tri_epoch / len(data_loader))
        loss_dict["invariance"]["validation"].append(loss_inv_epoch / len(data_loader))
        loss_dict["all"]["validation"].append(overall_loss / len(data_loader))


def run(args, k_fold):
    # read data
    if "musk1" in args.dataset:
        train_df = pd.read_csv(os.path.join("data",  arguments.dataset + ".data"), index_col=0, header=None)
        test_df = pd.read_csv(os.path.join("data",  arguments.dataset + "_test_" + str(k_fold) + ".data"), index_col=0, header=None)
        train_data_set = MuskDataset(train_df)
        test_data_set = MuskDataset(test_df)
    elif "elephant" in args.dataset or "fox" in args.dataset or "tiger" in args.dataset:
        train_df = pd.read_csv(os.path.join("data", arguments.dataset + "_train_" + str(k_fold) + ".data"))
        test_df = pd.read_csv(os.path.join("data", arguments.dataset + "_test_" + str(k_fold) + ".data"))
        train_data_set = AnimalDataset(train_df)
        test_data_set = AnimalDataset(test_df)
    elif "wiki" in args.dataset:
        if args.process_wiki:
            data_set = WikiDataset(df=os.path.join("data", "wiki.csv"), corp_path=r"data\wiki.json",
                                   label_path=r"data\label2index.pkl", category_path=r"data\category_counter.pkl")
        else:
            df = pd.read_csv("wiki.csv")
            data_set = WikiDataset(df=df, label_path=r"data\label2index.pkl", category_path=r"data\category_counter.pkl")
    else:
        print(f"Must Select Dataset")
        return

    with open("params.json", "r") as f:
        params = json.load(f)
    # data, model, optimizer and loss setup
    train_loader = torch.utils.data.DataLoader(train_data_set, params["batch_size"], shuffle=True, collate_fn=my_collate)
    val_loader = torch.utils.data.DataLoader(test_data_set, params["batch_size"], shuffle=True, collate_fn=my_collate)
    model = NeuralClusteringModel(class_num=params["class_num"], dim=train_data_set.df.shape[1], df=train_data_set.df).to(
        device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=params["l_r"])
    custom_loss = CustomDistanceLoss(batch_size=params["batch_size"])
    criterion = torch.nn.MSELoss(reduction="mean")
    model.normalize_cluster_center(params["alpha"])
    loss_dict = {key: {"train": [], "validation": [], "weight": [value]} for key, value in
                 params['loss_weights'].items()}
    loss_dict["all"] = {"train": [], "validation": []}
    params_tracker = {key: [value] for key, value in params["loss_weights"].items()}
    kde_tracker = {x: {"self": [], "other": []} for x in ["train", "validation"]}

    # Run experiment
    for epoch in range(params["epochs"]):
        # before train compute kde
        kde_tracker = update_kde(kde_tracker, model, train_data_set, test_data_set)
        train(model=model, data_loader=train_loader, optimizer=optimizer, criterion=criterion,
              custom_loss=custom_loss, epoch=epoch, data=train_data_set.data, loss_dict=loss_dict, params=params)
        validate(model=model, data_loader=val_loader, criterion=criterion, custom_loss=custom_loss,
                 epoch=epoch, data=test_data_set.data, loss_dict=loss_dict, params=params)
        print_losses(epoch, params["epochs"], loss_dict)
        # after train change losses and update - maybe not update for last?
        params_tracker = update_params(args, params_tracker, loss_dict, params)

    plot_data(args.run_label, loss_dict, params_tracker, kde_tracker, validation=True)
    save_data(args.run_label, loss_dict, model, train_data_set, params_tracker, kde_tracker, test_data_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='musk1', choices=['musk1', 'elephant', 'fox', 'tiger', 'wiki'])
    parser.add_argument("--run_label", type=str, default="musk1")
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--decay_weights', type=bool, default=False)
    parser.add_argument('--process_wiki', type=str, default=None)
    parser.add_argument('--fold', type=int, default=None)
    arguments = parser.parse_args()
    cnt = 1
    for dataset in ['musk1', 'elephant', 'fox', 'tiger']:
        arguments.dataset = dataset
        for param_to_change in ["reconstruction", "clustering", "contrastive", "triangle", "invariance"]:
            # if "musk1" in arguments.dataset:
            #     df = pd.read_csv(os.path.join("data", arguments.dataset + ".data"), index_col=0, header=None)
            # elif "elephant" in arguments.dataset or "fox" in arguments.dataset or "tiger" in arguments.dataset:
            #     df = pd.read_csv(os.path.join("data", arguments.dataset + ".data"))
            # # split df into 5
            # kf = KFold(n_splits=5, shuffle=True, random_state=42)
            # index_list = np.array(df.index.unique().tolist())
            # for fold, (train_index, test_index) in enumerate(kf.split(range(len(index_list)))):
            #     print(f"cnt is: {cnt}/125 | arguments.run_label is ", arguments.run_label)
            #     train_data = df.loc[index_list[train_index], :]
            #     test_data = df.loc[index_list[test_index], :]
            #     print(f"len of train data is: {train_data.shape} and test data: {test_data.shape}")
            #     print(f"index len: {test_data.shape} index: {test_data.index.unique()}")
            #     train_data.to_csv(os.path.join("data", arguments.dataset + "_train_" + str(fold) + ".data"))
            #     test_data.to_csv(os.path.join("data", arguments.dataset + "_test_" + str(fold) + ".data"))
            for fold in range(5):
                # print(f"arguments are: {arguments}")
                arguments.run_label = arguments.dataset + '_' + str(fold) + "_" + str(param_to_change).replace(".", "_")
                print(f"cnt is: {cnt}/100 | args is: {arguments.run_label}")
                change_params(param_to_change, 1)
                run(arguments, fold)
                change_params(param_to_change, 0)  # change back to zero
                cnt += 1
