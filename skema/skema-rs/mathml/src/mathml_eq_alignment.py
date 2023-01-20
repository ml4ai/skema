# -*- coding: utf-8 -*
import requests
import pydot
import numpy as np
from graspologic.match import graph_match

np.random.seed(1)
rng = np.random.default_rng(1)

op_dict = {"+": 1, "-": 2, "*": 3, "/": 4, "=": 5, "√": 6}


def generate_graph(file="seir_eq1.xml"):
    if '<math>' in file and '</math>' in file:
        content = file
    else:
        with open("../tests/" + file) as f:
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


def align_mathml_eqs(file1, file2):
    graph1 = generate_graph(file1)
    graph2 = generate_graph(file2)

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
    matching_ratio = round(1 - (num_diff_edges / num_edges), 2)

    long_len = len(node_labels1) if len(node_labels1) >= len(node_labels2) else len(node_labels2)
    aligned_indices1 = np.zeros((long_len)) - 1
    aligned_indices2 = np.zeros((long_len)) - 1
    for i in range(long_len):
        if i < len(node_labels1):
            if i in matched_indices1:
                aligned_indices1[i] = matched_indices2[np.where(matched_indices1 == i)[0][0]]
                aligned_indices2[matched_indices2[np.where(matched_indices1 == i)[0][0]]] = i

    return matching_ratio, node_labels1, node_labels2, aligned_indices1, aligned_indices2


if __name__ == "__main__":
    matching_ratio, node_labels1, node_labels2, aligned_indices1, aligned_indices2 = align_mathml_eqs("seir_eq1.xml", "seirdv_eq2.xml")

    print('matching ratio: ' + str(round(matching_ratio * 100, 2)) + '%')
    for i in range(len(node_labels1)):
        if aligned_indices1[i] != -1:
            print(str(node_labels1[i]) + '<=====>' + str(node_labels2[int(aligned_indices1[i])]))
        else:
            print(str(node_labels1[i]) + '<=====>missing')

    for i in range(len(node_labels2)):
        if i not in aligned_indices1:
            print('missing<=====>' + str(node_labels2[i]))

