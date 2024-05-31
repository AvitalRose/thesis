import os
import dcor
import pandas as pd
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_bag_to_bag_distance(bag_i, bag_j):
    return dcor.energy_distance(bag_i, bag_j)


def compute_bags_distance(bags_projections):
    energies_dict = {}
    for i, bag_i in enumerate(bags_projections):
        energies_dict[i] = {}
        for j, bag_j in enumerate(bags_projections):
            # compute energy distance for all bags
            energies_dict[i][j] = compute_bag_to_bag_distance(bag_i, bag_j)
    energies_df = pd.DataFrame.from_dict(energies_dict).values
    return energies_df


def update_params(args, params_tracker, loss_dict, params):
    if not args.decay_weights:
        return params_tracker
    # change params
    for loss_kind in loss_dict.keys():
        if loss_kind == "clustering" or loss_kind == "all":
            continue
        if loss_dict[loss_kind]["train"][-1] >= loss_dict[loss_kind]["train"][0]:
            params["loss_weights"][loss_kind] = params["loss_weights"][loss_kind] * (1 + params["xi"])
        else:
            params["loss_weights"][loss_kind] = 0.25 * 0.3 + params["loss_weights"][loss_kind] * (1 - 0.3)
    for loss_kind in loss_dict.keys():
        if loss_kind == "all":
            continue
        loss_dict[loss_kind]["weight"].append(params["loss_weights"][loss_kind])
    # params track
    for key, value in params["loss_weights"].items():
        params_tracker[key].append(value)
    return params_tracker


def update_kde(kde_tracker, model, train_set):
    x_projections = model.encode(torch.cat(train_set.X).to(device).float())[0]
    labels = torch.cat(train_set.label).tolist()
    y = torch.cat(train_set.data[1])
    data_df = pd.DataFrame(x_projections.detach().cpu().numpy(), columns=range(x_projections.shape[1]),
                           index=y.tolist())
    data_df["label"] = labels
    ratio_self, ratio_other = compute_kde(data_df)  # change this for wiki
    kde_tracker["self"].append(ratio_self)
    kde_tracker["other"].append(ratio_other)
    return kde_tracker


def compute_kde(df):
    x_list, label_list, _ = bags_aggregate(df)
    energies = pd.DataFrame(compute_bags_distance(x_list))
    # analysis on data by yoram measurement
    analysis = pd.DataFrame(group2groupdist(energies, label_list, None))
    return analysis.loc["ratio"].values


def compute_measurement(df, sigma):
    # for each point define the kde from it onward
    df = df.apply(lambda x: np.exp((-x**2) / (2 * sigma)))
    df["density"] = (df.sum(axis=1) - 1) / (df.shape[0] - 1)  # ignoring volumes
    return df["density"].mean()


def group2groupdist(projections, labels, index_to_label):
    measure_dict = {}
    combined_df = projections.copy()
    combined_df.index = [i[0] for i in labels]
    combined_df.columns = combined_df.index
    s = combined_df.stack().std() / 10
    for i in set(combined_df.index):
        name = index_to_label[i] if index_to_label else i
        if len(combined_df.loc[i, i].shape) < 2:
            continue
        measure_dict[str(name)] = {}
        # self 2 self
        measure_dict[str(name)]["self2self"] = compute_measurement(combined_df.loc[i, i].copy(), s)
        # self 2 other
        measure_dict[str(name)]["self2other"] = compute_measurement(combined_df.loc[i, combined_df.columns != i].copy(), s)
    df = pd.DataFrame(measure_dict).stack().to_frame()
    # to break out the lists into columns
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    af = df.stack().unstack(level=1).unstack(level=1)
    af.loc["ratio"] = af.loc["self2self"] / af.loc["self2other"]
    return af


def bags_aggregate(df):
    X = []
    label_list = []
    y = []
    molc = list(set(list(df.index)))
    molc_dict = {m: i for i, m in enumerate(molc)}
    for m in molc:
        bag = df[df.index == m]
        label = bag['label'].values
        X.append(torch.tensor(bag.loc[:, df.columns != "label"].values))
        label_list.append(label)
        y_label = molc_dict[m]
        y.append(torch.tensor([y_label]*bag.values.shape[0]))
    return X, label_list, y


def print_losses(epoch, epochs, loss_dict):
    # output = f"Epoch"
    output = f"[{epoch}/{epochs}]:"
    output += f"T: "
    for loss_name, loss in loss_dict.items():
        output += f"{loss_name[:3]}: {loss['train'][-1]:.2f} "
    if len(loss_dict["clustering"]["validation"]) > 0:
        output += f"\tV: "
        for loss_name, loss in loss_dict.items():
            output += f"{loss_name[:3]}: {loss['validation'][-1]:.2f} "
    print(output)


def generic_plot_dict_losses(loss_dict, label, num_plot=4, plot_val=False):
    if num_plot == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        row_i = 2
        col_i = 2
    elif num_plot == 6:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
        row_i = 3
        col_i = 3
    fig.suptitle(f"Losses vs Epochs\n{label}")
    i = 0
    colors = ["b", "g", "r", "m", "orange", "pink"]
    for key in loss_dict.keys():
        if key == "inv":
            continue
        print(f"len of epochs is: {len(loss_dict[key]['train'])}")
        axs[int(i / row_i), i % col_i].plot(loss_dict[key]["train"], label="training", color=colors[i])
        if plot_val:
            axs[int(i / row_i), i % col_i].plot(loss_dict[key]["validation"][1:], label="validation", color="black")
        if int(i / row_i) == 1:
            axs[int(i / row_i), i % col_i].set_xlabel("epochs", fontsize=10)
        if i % col_i == 0:
            axs[int(i / row_i), i % col_i].set_ylabel("loss", fontsize=10)
        # axs[int(i / 2), i % 2].set_xticks(range(len(loss_dict[key])))
        axs[int(i / row_i), i % col_i].margins(0)
        axs[int(i / row_i), i % col_i].set_title(f"{key}", fontsize=10)
        axs[int(i / row_i), i % col_i].legend()
        # axs[int(i / 2), i % 2].grid(True)
        i += 1
    # plt.savefig(f"loss vs epochs" +
                # str(datetime.datetime.now()).split(" ")[0].replace("-", "_").replace(":", "_")
                # .replace(" ", ""))
    plt.savefig(os.path.join("results", f"losses vs epochs " + str(label.split("W")[0])))
    # plt.show()
    plt.clf()

    # all plots on same figure
    for i, (key, value) in enumerate(loss_dict.items()):
        plt.plot(value["train"], label=key, color=colors[i])
    plt.legend()
    plt.margins(0)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.ylabel("Losses")
    plt.title(f"All Losses vs Epochs")
    # plt.savefig(f"All losses" + str(label.split("W")[0]))
    # plt.savefig(f"All losses " +
                # str(datetime.datetime.now()).split(" ")[0].replace("-", "_").replace(":", "_")
                # .replace(" ", ""))
    # plt.show()
    plt.clf()


def generic_plot_dictionary(dictionary, label):
    for key, value in dictionary.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.title(label)
    plt.xlabel("Epochs")
    plt.ylabel(label)
    plt.title(f"{label} vs Epochs")
    plt.savefig(os.path.join("results", label))
    # plt.show()
    plt.clf()


def plot_data(run_label, loss_dict, params_tracker, kde_tracker):
    # plot losses
    generic_plot_dict_losses(loss_dict, run_label, num_plot=6, plot_val=True)

    # plot params
    generic_plot_dictionary(params_tracker, "loss weights " + str(run_label))

    # plot kde
    generic_plot_dictionary(kde_tracker, "kde track " + str(run_label))


def save_data(run_label, loss_dict, model, data_set, params_tracker, kde_tracker):
    all_results_dict = {
        "loss_dict": loss_dict,
        "model": model,
        "data_set": data_set,
        "params_tracker": params_tracker,
        "kde_tracker": kde_tracker
    }
    # saving pkl file of loss dict
    with open(os.path.join("results", str(run_label) + "results_dict.pickle"), "wb") as output_file:
        pkl.dump(all_results_dict, output_file)

