import os
import json
import pickle as pkl
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from draw_struct import draw_networkx_nodes_ellipses

simdial_clean = [
    "bus-CleanSpec-2000", "movie-CleanSpec-2000",
    "restaurant_style-CleanSpec-2000", "restaurant-CleanSpec-2000",
    "weather-CleanSpec-2000"
]
simdial_mix = [
    "bus-MixSpec-2000", "movie-MixSpec-2000", "restaurant_style-MixSpec-2000",
    "restaurant-MixSpec-2000", "weather-MixSpec-2000"
]
simdial_json_list = simdial_clean  # + simdial_mix

dial2mix = [
    "bus-CleanSpec-2000",
    "movie-CleanSpec-2000",
    "restaurant-CleanSpec-2000",
]
mix_data = {"train": [], "test": []}

for simdial_json in simdial_json_list:
    with open(os.path.join("data/simdial", simdial_json + ".json"), "r") as f:
        data = json.loads(f.read())
        dialogs = data["dialogs"]
        cnt = 0
        train_dial = []
        test_dial = []
        act_lists = []
        act_set = []
        for dialog in dialogs:
            last_speaker = 'SYS'
            dial_tmp = []
            utt_pair = []
            act_list = []
            act = ""
            for turn in dialog:
                for a in turn['actions']:
                    act += a["act"]
                    for i in range(len(a["parameters"])):
                        if type(a["parameters"][i]) is dict:
                            act += "".join(
                                [str(x) for x in a["parameters"][i].keys()])
                        else:
                            for item in a["parameters"][i]:
                                if type(item) is list:
                                    act += item[0]
                                elif type(item) is str and item[0] == "#":
                                    act += item
                    act += "-"
                if utt_pair == []:
                    utt_pair = [cnt, turn['utt']]
                    act += "||"
                else:
                    utt_pair.append(turn['utt'])
                    utt_pair.append(act)
                    act_list.append(act)
                    act_set.append(act)
                    dial_tmp.append(tuple(utt_pair))
                    act = ""
                    utt_pair = []
            if utt_pair != []:
                utt_pair.append("<SILENCE>")
                utt_pair.append(act)
                act_list.append(act)
                act_set.append(act)
                dial_tmp.append(tuple(utt_pair))
            if cnt < 1600:
                train_dial.append(dial_tmp)
            else:
                test_dial.append(dial_tmp)
            act_lists.append(act_list)
            cnt += 1
    act_set = set(act_set)
    act_set = list(act_set)

    trans_cnt = np.zeros((len(act_set), len(act_set)))
    for i in range(len(act_lists)):
        for j in range(len(act_lists[i]) - 1):
            from_idx = act_set.index(act_lists[i][j])
            to_idx = act_set.index(act_lists[i][j + 1])
            trans_cnt[from_idx, to_idx] += 1
    trans_prob = np.zeros((len(act_set), len(act_set)))
    for i in range(len(act_set)):
        if trans_cnt[i].sum() != 0:
            trans_prob[i] = trans_cnt[i] / trans_cnt[i].sum()

    #  draw the interpretion with networkx and matplotlib
    G = nx.DiGraph()
    node_labels = {}
    for i in range(len(act_set)):
        G.add_node(i)
        node_labels[i] = act_set[i]

    edge_labels = {}
    for i in range(len(act_set)):
        for j in range(len(act_set)):
            if trans_prob[i, j] > 0.0:
                G.add_edge(i, j)
                edge_labels[(i, j)] = "%.2f" % trans_prob[i, j]

    pos = nx.spring_layout(G)
    node_width = [5 * len(node_labels[node]) for node in G.nodes()]
    draw_networkx_nodes_ellipses(G,
                                 pos=pos,
                                 node_width=node_width,
                                 node_height=10,
                                 node_color='w',
                                 edge_color='k',
                                 alpha=0.0)

    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=7)
    nx.draw_networkx_edges(G, pos=pos, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    # plt.show()
    fig.savefig(os.path.join("data/simdial", simdial_json + ".png"), dpi=100)

    data_to_write = {
        "train": train_dial,
        "test": test_dial,
        "trans_prob": trans_prob,
        "act_list": act_set
    }
    with open(os.path.join("data/simdial", simdial_json + ".pkl"), 'wb') as f:
        pkl.dump(data_to_write, f)

    if simdial_json in dial2mix:
        mix_data["train"] += train_dial
        mix_data["test"] += test_dial

random.shuffle(mix_data["train"])
random.shuffle(mix_data["test"])

with open(os.path.join("data/simdial", "-".join(dial2mix) + ".pkl"),
          'wb') as f:
    pkl.dump(mix_data, f)
