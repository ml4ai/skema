# -*- coding: utf-8 -*
"""
All the functions required by performing incremental structure alignment (ISA)
Author: Liang Zhang (liangzh@arizona.edu)
Updated date: March 3, 2023
"""

import warnings

from typing import List, Tuple, Any, Union

from numpy import ndarray
from pydot import Dot

warnings.filterwarnings('ignore')
import requests
import pydot
import numpy as np
from graspologic.match import graph_match
from graphviz import Source
import graphviz
from copy import deepcopy

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


def has_edge(dot: pydot.Dot, src: str, dst: str) -> bool:
    """
    Check if an edge exists between two nodes in a PyDot graph object.

    Args:
        dot (pydot.Dot): PyDot graph object.
        src (str): Source node ID.
        dst (str): Destination node ID.

    Returns:
        bool: True if an edge exists between src and dst, False otherwise.
    """
    edges = dot.get_edges()
    for edge in edges:
        if edge.get_source() == src and edge.get_destination() == dst:
            return True
    return False


'''
return the union graph for visualizing the alignment results
input: 
output: dot graph
'''


def get_union_graph(graph1: pydot.Dot, graph2: pydot.Dot,
                    aligned_idx1: List[int], aligned_idx2: List[int]) -> pydot.Dot:
    g2idx2g1idx = {str(x): str(-1) for x in range(len(graph2.get_nodes()))}
    union_graph = deepcopy(graph1)
    '''
    set the aligned variables or terms as a blue circle; 
    if their names are the same, show one name; 
    if not, show two names' connection using '<<|>>'
    '''
    for i in range(len(aligned_idx1)):
        if union_graph.get_nodes()[aligned_idx1[i]].obj_dict['attributes']['label'].replace('"', '').lower() != \
                graph2.get_nodes()[aligned_idx2[i]].obj_dict['attributes']['label'].replace('"', '').lower():
            union_graph.get_nodes()[aligned_idx1[i]].obj_dict['attributes']['label'] = \
                union_graph.get_nodes()[aligned_idx1[i]].obj_dict['attributes']['label'].replace('"',
                                                                                                 '') + ' <<|>> ' + \
                graph2.get_nodes()[aligned_idx2[i]].obj_dict['attributes']['label'].replace('"', '')

        union_graph.get_nodes()[aligned_idx1[i]].obj_dict['attributes']['color'] = 'blue'
        g2idx2g1idx[str(aligned_idx2[i])] = str(aligned_idx1[i])

    # represent the nodes only in graph 1 as a red circle
    for i in range(len(union_graph.get_nodes())):
        if i not in aligned_idx1:
            union_graph.get_nodes()[i].obj_dict['attributes']['color'] = 'red'

    # represent the nodes only in graph 2 as a green circle
    for i in range(len(graph2.get_nodes())):
        if i not in aligned_idx2:
            graph2.get_nodes()[i].obj_dict['attributes']['color'] = 'green'
            graph2.get_nodes()[i].obj_dict['name'] = str(len(union_graph.get_nodes()))
            union_graph.add_node(graph2.get_nodes()[i])
            g2idx2g1idx[str(i)] = str(len(union_graph.get_nodes()) - 1)

    # add the edges of graph 2 to graph 1
    for edge in union_graph.get_edges():
        edge.obj_dict['attributes']['color'] = 'red'

    for edge in graph2.get_edges():
        x, y = edge.obj_dict['points']
        if has_edge(union_graph, g2idx2g1idx[x], g2idx2g1idx[y]):
            if union_graph.get_edge(g2idx2g1idx[x], g2idx2g1idx[y])[0].obj_dict['attributes']['label'].lower() == \
                    edge.obj_dict['attributes']['label'].lower():
                union_graph.get_edge(g2idx2g1idx[x], g2idx2g1idx[y])[0].obj_dict['attributes']['color'] = 'blue'
            else:
                e = pydot.Edge(g2idx2g1idx[x], g2idx2g1idx[y], label=edge.obj_dict['attributes']['label'],
                               color='green')
                union_graph.add_edge(e)
        else:
            e = pydot.Edge(g2idx2g1idx[x], g2idx2g1idx[y], label=edge.obj_dict['attributes']['label'],
                           color='green')
            union_graph.add_edge(e)

    return union_graph


def check_square_array(arr: np.ndarray) -> List[int]:
    """
    Given a square numpy array, returns a list of size equal to the length of the array,
    where each element of the list is either 0 or 1, depending on whether the corresponding
    row and column of the input array are all 0s or not.

    Parameters:
    arr (np.ndarray): a square numpy array

    Returns:
    List[int]: a list of 0s and 1s
    """

    n = arr.shape[0]  # get the size of the array
    result = []
    for i in range(n):
        # Check if the ith row and ith column are all 0s
        if np.all(arr[i, :] == 0) and np.all(arr[:, i] == 0):
            result.append(0)  # if so, append 0 to the result list
        else:
            result.append(1)  # otherwise, append 1 to the result list
    return result


'''
align two equation graphs using the seeded graph matching (SGD) algorithm [1].

[1] Fishkind, D. E., Adali, S., Patsolic, H. G., Meng, L., Singh, D., Lyzinski, V., & Priebe, C. E. (2019). 
Seeded graph matching. Pattern recognition, 87, 203-215.

Input: the paths of the two equation MathMLs; mode 0: without considering any priors; mode 1: having a heuristic prior 
with the similarity of node labels; mode 2: TBD
Output:
    matching_ratio: the matching ratio between the equations 1 and the equation 2
    num_diff_edges: the number of different edges between the equations 1 and the equation 2
    node_labels1: the name list of the variables and terms in the equation 1
    node_labels2: the name list of the variables and terms in the equation 2
    aligned_indices1: the aligned indices in the name list of the equation 1
    aligned_indices2: the aligned indices in the name list of the equation 2
'''


def align_mathml_eqs(file1: str = "", file2: str = "", mode: int = 1) \
        -> Tuple[Any, Any, List[str], List[str], Union[int, Any], Union[int, Any], Dot, List[int]]:
    graph1 = generate_graph(file1)
    graph2 = generate_graph(file2)

    amatrix1, node_labels1 = generate_amatrix(graph1)
    amatrix2, node_labels2 = generate_amatrix(graph2)

    if mode == 0:
        seed1 = []
        seed2 = []
    elif mode == 1:
        seed1, seed2 = get_seeds(node_labels1, node_labels2)
    else:
        seed1 = []
        seed2 = []

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
    perfectly_matched_indices1 = check_square_array(diff_edges)
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

    union_graph = get_union_graph(graph1, graph2, [int(i) for i in matched_indices1.tolist()],
                                  [int(i) for i in matched_indices2.tolist()])

    return matching_ratio, num_diff_edges, node_labels1, node_labels2, aligned_indices1, aligned_indices2, union_graph, perfectly_matched_indices1
