# -*- coding: utf-8 -*
import requests
import pydot
import numpy as np
from graspologic.match import graph_match

np.random.seed(1)
rng = np.random.default_rng(1)

op_dict = {"+": 1, "-": 2, "*": 3, "/": 4, "=": 5, "√": 6}


def generate_graph(filename="seir_eq1.xml"):
    with open("../tests/" + filename) as f:
        content = f.read()

    digraph = requests.put('http://localhost:8080/mathml/math-exp-graph', data=content.encode('utf-8'))
    graph = pydot.graph_from_dot_data(str(digraph.text))[0]
    return graph


def generate_amatrix(graph):
    node_labels = []
    for node in graph.get_nodes():
        node_labels.append(node.obj_dict['attributes']['label'].replace('"', ''))

    amatrix = np.zeros((len(node_labels), len(node_labels)))

    for edge in graph.get_edges():
        x, y = edge.obj_dict['points']
        label = edge.obj_dict['attributes']['label'].replace('"', '')
        amatrix[int(x)][int(y)] = op_dict[label] if label in op_dict else 7

    return amatrix, node_labels


if __name__ == "__main__":
    graph1 = generate_graph("seir_eq1.xml")
    graph2 = generate_graph("seirdv_eq2.xml")

    amatrix1, node_labels1 = generate_amatrix(graph1)
    amatrix2, node_labels2 = generate_amatrix(graph2)

    seed1 = [0, 1]
    seed2 = [0, 1]
    for i in range(2, len(node_labels1)):
        for j in range(2, len(node_labels2)):
            if node_labels1[i].lower() == node_labels2[j].lower():
                seed1.append(i)
                seed2.append(j)

    partial_match = np.column_stack((seed1, seed2))

    matched_indices1, matched_indices2, _, _ = graph_match(
        amatrix1, amatrix2, partial_match=partial_match, padding="adopted", rng=rng, max_iter=50
    )

    big_graph_idx = 0 if len(node_labels1) >= len(node_labels2) else 1
    if big_graph_idx == 0:
        big_graph = amatrix1
        big_graph_matched_indices = matched_indices1
        small_graph = amatrix2
        small_graph_matched_indices = matched_indices2
    else:
        big_graph = amatrix2
        big_graph_matched_indices = matched_indices2
        small_graph = amatrix1
        small_graph_matched_indices = matched_indices1

    small_graph_aligned = small_graph[small_graph_matched_indices][:, small_graph_matched_indices]
    small_graph_aligned_full = np.zeros(big_graph.shape)
    small_graph_aligned_full[np.ix_(big_graph_matched_indices, big_graph_matched_indices)] = small_graph_aligned

    num_edges = ((big_graph + small_graph_aligned_full) > 0).sum()
    diff_edges = abs(big_graph - small_graph_aligned_full)
    diff_edges[diff_edges > 0] = 1
    num_diff_edges = np.sum(diff_edges)
    matching_ratio = 1 - (num_diff_edges / num_edges)
    print('matching ratio: ' + str(round(matching_ratio * 100, 2)) + '%')


    long_len = len(node_labels1) if len(node_labels1) >= len(node_labels2) else len(node_labels2)

    for i in range(long_len):
        if i < len(node_labels1):
            if i not in matched_indices1:
                print(str(node_labels1[i]) + '<=====>missing')
            else:
                print(str(node_labels1[i]) + '<=====>' + str(
                    node_labels2[matched_indices2[np.where(matched_indices1 == i)[0][0]]]))
        else:
            print('missing<=====>' + str(node_labels2[i]))
