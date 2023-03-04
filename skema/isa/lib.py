# -*- coding: utf-8 -*
"""
All the functions required by performing incremental structure alignment (ISA)
Author: Liang Zhang (liangzh@arizona.edu)
Updated date: March 3, 2023
"""

import warnings

from typing import List, Tuple, Any

from numpy import ndarray

warnings.filterwarnings('ignore')
import requests
import pydot
import numpy as np
from graspologic.match import graph_match
from graphviz import Source

# Set up the random seed
np.random.seed(1)
rng = np.random.default_rng(1)

# The encodings of basic operators when converting adjacency matrix
op_dict = {"+": 1, "-": 2, "*": 3, "/": 4, "=": 5, "√": 6}

'''
Call the REST API of math-exp-graph to convert the MathML input to its GraphViz representation
Ensure running the REST API before calling this function
Input: file directory or the MathML string
Output: the GraphViz representation (pydot.Dot)
'''


def generate_graph(file: str = "", render: bool = False) -> pydot.Dot:
    if '<math>' in file and '</math>' in file:
        content = file
    else:
        with open(file) as f:
            content = f.read()

    digraph = requests.put('http://localhost:8080/mathml/math-exp-graph', data=content.encode('utf-8'))
    if render:
        src = Source(digraph.text)
        src.render('doctest-output/mathml_exp_tree', view=True)
    graph = pydot.graph_from_dot_data(str(digraph.text))[0]
    return graph


'''
Convert the GraphViz representation to its corresponding adjacency matrix
Input: the GraphViz representation
Output: the adjacency matrix and the list of the names of variables and terms appeared in the expression  
'''


def generate_amatrix(graph: pydot.Dot) -> Tuple[ndarray, List[str]]:
    node_labels = []
    for node in graph.get_nodes():
        node_labels.append(node.obj_dict['attributes']['label'].replace('"', ''))

    amatrix = np.zeros((len(node_labels), len(node_labels)))

    for edge in graph.get_edges():
        x, y = edge.obj_dict['points']
        label = edge.obj_dict['attributes']['label'].replace('"', '')
        amatrix[int(x)][int(y)] = op_dict[label] if label in op_dict else 7

    return amatrix, node_labels


'''
Calculate the seeds in the two equations
Input: the name lists of the variables and terms in the equation 1 and the equation 2
Output: the seed indices from the equations 1 and the equation 2
'''


def get_seeds(node_labels1: List[str], node_labels2: List[str]) -> Tuple[List[int], List[int]]:
    seed1 = [0, 1]
    seed2 = [0, 1]
    for i in range(2, len(node_labels1)):
        for j in range(2, len(node_labels2)):
            if node_labels1[i].lower() == node_labels2[j].lower():
                seed1.append(i)
                seed2.append(j)

    return seed1, seed2


'''
align two equation graphs using the seeded graph matching (SGD) algorithm [1].

[1] Fishkind, D. E., Adali, S., Patsolic, H. G., Meng, L., Singh, D., Lyzinski, V., & Priebe, C. E. (2019). 
Seeded graph matching. Pattern recognition, 87, 203-215.

Input: the paths of the two equation MathMLs
Output:
    matching_ratio: the match ratio between the equations 1 and the equation 2
    num_diff_edges: the number of different edges between the equations 1 and the equation 2
    node_labels1: the name list of the variables and terms in the equation 1
    node_labels2: the name list of the variables and terms in the equation 2
    aligned_indices1: the aligned indices in the name list of the equation 1
    aligned_indices2: the aligned indices in the name list of the equation 2
'''


def align_mathml_eqs(file1: str = "", file2: str = "") \
        -> Tuple[Any, ndarray, List[str], List[str], Any, Any]:
    graph1 = generate_graph(file1)
    graph2 = generate_graph(file2)

    amatrix1, node_labels1 = generate_amatrix(graph1)
    amatrix2, node_labels2 = generate_amatrix(graph2)

    seed1, seed2 = get_seeds(node_labels1, node_labels2)
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

    return matching_ratio, num_diff_edges, node_labels1, node_labels2, aligned_indices1, aligned_indices2
