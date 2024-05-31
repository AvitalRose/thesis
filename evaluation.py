import os
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import string

from matplotlib.collections import PatchCollection


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width / len(orig_handle.colors) * i - handlebox.xdescent,
                                          -handlebox.ydescent], width / len(orig_handle.colors), height,
                                         facecolor=c, edgecolor='none'))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


def collect_kde(kde_path=r"C:\Users\avita\Documents\skl\yoram_lab\graphics\kde_plots"):
    files_kde = os.listdir(kde_path)
    plots_by_loss = {}
    plots_by_dataset = {}
    for file in files_kde:
        with open(os.path.join(kde_path, file), 'rb') as f:
            data = pkl.load(f)
        loss_type = file.split("_")[1].split(" .pkl")[0]
        dataset_type = file.split("_")[0].split(" ")[-1]
        if loss_type not in plots_by_loss:
            plots_by_loss[loss_type] = {}
        if dataset_type not in plots_by_dataset:
            plots_by_dataset[dataset_type] = {}
        plots_by_loss[loss_type][dataset_type] = {"0": data["self"], "1": data["other"]}
        plots_by_dataset[dataset_type][loss_type] = {"0": data["self"], "1": data["other"]}
    # plot_kde_box_plot(plots_by_loss)
    # for key in plots_by_dataset.keys():
    #     plot_kde_by_dataset(key, plots_by_dataset[key])
    return plots_by_loss, plots_by_dataset


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def plot_kde_box_plot(loss_dict):
    my_dict = {"0": {}, "1": {}}
    for loss_type, dataset_dict in loss_dict.items():  # box plot for each loss
        for dataset_type, dataset_kde_list in dataset_dict.items():
            for label_type, label_list in dataset_kde_list.items():
                if label_type not in my_dict.keys():
                    my_dict[label_type] = {}
                if loss_type not in my_dict[label_type].keys():
                    my_dict[label_type][loss_type] = []
                my_dict[label_type][loss_type].append(np.log(label_list[-1] / label_list[0]))
    data_a = [i for i in my_dict["0"].values()]
    data_b = [i for i in my_dict["1"].values()]

    ticks = [string.capwords(i) for i in(loss_dict.keys())]

    plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"

    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a))) * 2.0 - 0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C')  # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Label 0')
    plt.plot([], c='#2C7BB6', label='Label 1')
    plt.axhline(y=0, color='black', lw=1)
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.title("KDE Improvement For Single Loss")
    plt.xlabel("Loss Used In Training", labelpad=10)
    plt.ylabel("Log Ratio Of KDE Improvement")
    plt.tight_layout()
    plt.show()


def plot_kde_box_plot_wiki(loss_dict):
    my_dict = {}
    for loss_type, label_dict in loss_dict.items():  # box plot for each loss
        for label_type, label_list in label_dict.items():
            if loss_type not in my_dict.keys():
                my_dict[loss_type] = []
            my_dict[loss_type].append(np.log(label_list[-1] / label_list[0]))

    ticks = [string.capwords(i) for i in loss_dict.keys()]
    plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    # color_list = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
    # color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    plt.boxplot(my_dict.values())
    plt.axhline(y=0, color="black")

    plt.legend()
    plt.xticks([1, 2], ticks)
    plt.title("KDE Improvement For Single Loss On Wikipedia")
    plt.xlabel("Loss Used In Training", labelpad=10)
    plt.ylabel("Log Ratio Of KDE Improvement")
    plt.tight_layout()
    plt.show()


def plot_kde_by_dataset(dict_name, loss_dict):
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    # colors = ["r", "g", "b", "m"]
    colors = [("#a6cee3", "#1f78b4"), ("#b2df8a", "#33a02c"), ("#fb9a99", "#e31a1c"), ("#fdbf6f", "#ff7f00")]
    line_styles = ["solid", "dashed"]
    for i, (loss_type, value) in enumerate(loss_dict.items()):
        for j, label_type in enumerate(value.keys()):
            plt.plot(value[label_type], color=colors[i][int(label_type)], linestyle=line_styles[j], linewidth=1)
        # plt.plot([], label=loss_type, color=colors[i])
    # plt.legend()
    # ------ get the legend-entries that are already attached to the axis
    h, l = ax.get_legend_handles_labels()
    # ------ append the multicolor legend patches
    for i, key in enumerate(loss_dict.keys()):
        h.append(MulticolorPatch(list(colors[i])))
        l.append(string.capwords(key))
    # ------ create the legend
    fig.legend(h, l, handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc='right',bbox_to_anchor=[0.9, 0.2])
    plt.title(f"KDE By Isolate Losses For {string.capwords(dict_name)}")
    plt.xlabel("Epoch")
    plt.ylabel("KDE")
    plt.show()


def change_wiki_pkl(wiki_kde_path=r"C:\Users\avita\Documents\skl\yoram_lab\kde track ID wiki_reconstruction .pkl"):
    with open(wiki_kde_path, "rb") as f:
        d = pkl.load(f)
    wiki_dict = {i: [] for i in range(len(d["all"][0]))}
    for i, epoch in enumerate(d["all"]):
        for label, value in enumerate(epoch):
            wiki_dict[label].append(value)
    return wiki_dict


def bar_plot_kde(loss_dict):
    my_dict = {"0": {}, "1": {}}
    plt.rcParams["font.family"] = "Times New Roman"
    for dataset_type, loss_value_dict in loss_dict.items():  # box plot for each loss
        for loss_type, dataset_kde_list in loss_value_dict.items():
            for label_type, label_list in dataset_kde_list.items():
                if label_type not in my_dict.keys():
                    my_dict[label_type] = {}
                if dataset_type not in my_dict[label_type].keys():
                    my_dict[label_type][dataset_type] = []
                my_dict[label_type][dataset_type].append(round(label_list[-1] - label_list[0], 2))
    species = [list(loss_dict[i].keys()) for i in loss_dict.keys()][0]
    x = np.arange(len(species))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()
    colors = [("#a6cee3", "#1f78b4"), ("#b2df8a", "#33a02c"), ("#fb9a99", "#e31a1c"), ("#fdbf6f", "#ff7f00")]
    for i, attribute in enumerate(my_dict["0"].keys()):
        offset = width * multiplier
        rects = ax.bar(x + offset, my_dict["0"][attribute], width, color=colors[i][1], edgecolor="black")
        rects1 = ax.bar(x + offset, my_dict["1"][attribute], width, bottom=my_dict["0"][attribute], color=colors[i][0],
                        edgecolor="black")
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # ------ get the legend-entries that are already attached to the axis
    h, l = ax.get_legend_handles_labels()
    # ------ append the multicolor legend patches
    for i, key in enumerate(my_dict["0"].keys()):
        h.append(MulticolorPatch(list(colors[i])))
        l.append(string.capwords(key))

    # ------ create the legend
    fig.legend(h, l, handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc='lower right',
               bbox_to_anchor=[0.9, 0.2])  # this will change once the data changes

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('KDE Difference Between Before & After')
    ax.set_xlabel('Single Losses Used')
    ax.set_title('KDE Difference For Single Losses', pad=20)
    ax.set_xticks(x + width, species)
    ax.axhline(y=0, color='black', lw=1)
    plt.show()


def bar_plot_kde_wiki(loss_dict):
    plt.rcParams["font.family"] = "Times New Roman"
    my_dict = {}
    for loss_type, loss_value_dict in loss_dict.items():  # box plot for each loss
        for label_type, label_list in loss_value_dict.items():
            if label_type not in my_dict.keys():
                my_dict[label_type] = []
            my_dict[label_type].append(round(label_list[-1] - label_list[0], 2))
    my_dict = {key: tuple(value) for key, value in my_dict.items()}
    species = list(loss_dict.keys())
    x = np.arange(len(species))  # the label locations
    width = 0.05  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
    for i, attribute in enumerate(my_dict.keys()):
        offset = width * multiplier
        rects = ax.bar(x + offset, my_dict[attribute], width, edgecolor="black", label="label " + str(attribute), color=colors[i])
        multiplier += 1

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('KDE Difference Between Before & After')
    ax.set_xlabel('Single Losses Used')
    ax.set_title('KDE Difference For Single Losses', pad=20)
    ax.set_xticks(x + width, species)
    ax.axhline(y=0, color='black', lw=1)
    ax.legend()
    plt.show()


def plot_loss_for_all_single_losses(files_path=r"C:\Users\avita\Documents\skl\yoram_lab\graphics\loss dict"):
    # get file names
    files_list = [os.path.join(files_path, i) for i in os.listdir(files_path)]
    # plot
    plt.rcParams["font.family"] = "Times New Roman"
    colors = [("#a6cee3", "#1f78b4"), ("#b2df8a", "#33a02c"), ("#fb9a99", "#e31a1c"), ("#fdbf6f", "#ff7f00")]
    fig, ax = plt.subplots()
    h, l = ax.get_legend_handles_labels()
    for i, file_name in enumerate(files_list):
        loss_name = file_name.split("loss_dict.pickle")[0].split("_")[-1]
        with open(file_name, 'rb') as f:
            loss_dict = pkl.load(f)
        plt.plot(loss_dict[loss_name]['train'], color=colors[i][0])
        plt.plot(loss_dict[loss_name]['validation'], color=colors[i][1])
        h.append(MulticolorPatch(list(colors[i])))
        l.append(string.capwords(file_name.split("loss_dict.pickle")[0].split("_")[-1]))

    fig.legend(h, l, handler_map={MulticolorPatch: MulticolorPatchHandler()}, loc='lower right',
               bbox_to_anchor=[0.9, 0.2])  # this will change once the data changes
    plt.title("Loss Over Time For Single Losses")
    plt.xlabel("Epochs", labelpad=10)
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # loss_d, dataset_d = collect_kde()
    # wiki_dictionary = change_wiki_pkl()
    # wiki_dictionary1 = change_wiki_pkl()
    # plots_by_loss = {"rec0": wiki_dictionary, "rec1": wiki_dictionary1}
    # plot_kde_box_plot_wiki(plots_by_loss)
    # bar_plot_kde_wiki(plots_by_loss)
    plot_loss_for_all_single_losses()

