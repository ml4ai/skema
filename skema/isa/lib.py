# -*- coding: utf-8 -*
"""
All the functions required by performing incremental structure alignment (ISA)
Author: Liang Zhang (liangzh@arizona.edu)
Updated date: August 24, 2023
"""
import json
import warnings
from typing import List, Any, Union, Dict
from numpy import ndarray
from pydot import Dot

warnings.filterwarnings("ignore")
import requests
import pydot
import numpy as np
from graspologic.match import graph_match
from graphviz import Source
import graphviz
from copy import deepcopy
import Levenshtein
from typing import Tuple
import re
import xml.etree.ElementTree as ET
import html
from sentence_transformers import SentenceTransformer, util

# Set up the random seed
np.random.seed(4)
rng = np.random.default_rng(4)

# The encodings of basic operators when converting adjacency matrix
op_dict = {"+": 1, "-": 2, "*": 3, "/": 4, "=": 5, "√": 6}

# Greek letters mapping
# List of Greek letters mapping to their lowercase, name, and Unicode representation
greek_letters: List[List[str]] = [
    ["α", "alpha", "&#x03B1;"],
    ["β", "beta", "&#x03B2;"],
    ["γ", "gamma", "&#x03B3;"],
    ["δ", "delta", "&#x03B4;"],
    ["ε", "epsilon", "&#x03B5;"],
    ["ζ", "zeta", "&#x03B6;"],
    ["η", "eta", "&#x03B7;"],
    ["θ", "theta", "&#x03B8;"],
    ["ι", "iota", "&#x03B9;"],
    ["κ", "kappa", "&#x03BA;"],
    ["λ", "lambda", "&#x03BB;"],
    ["μ", "mu", "&#x03BC;"],
    ["ν", "nu", "&#x03BD;"],
    ["ξ", "xi", "&#x03BE;"],
    ["ο", "omicron", "&#x03BF;"],
    ["π", "pi", "&#x03C0;"],
    ["ρ", "rho", "&#x03C1;"],
    ["σ", "sigma", "&#x03C3;"],
    ["τ", "tau", "&#x03C4;"],
    ["υ", "upsilon", "&#x03C5;"],
    ["φ", "phi", "&#x03C6;"],
    ["χ", "chi", "&#x03C7;"],
    ["ψ", "psi", "&#x03C8;"],
    ["ω", "omega", "&#x03C9;"],
    ["Α", "Alpha", "&#x0391;"],
    ["Β", "Beta", "&#x0392;"],
    ["Γ", "Gamma", "&#x0393;"],
    ["Δ", "Delta", "&#x0394;"],
    ["Ε", "Epsilon", "&#x0395;"],
    ["Ζ", "Zeta", "&#x0396;"],
    ["Η", "Eta", "&#x0397;"],
    ["Θ", "Theta", "&#x0398;"],
    ["Ι", "Iota", "&#x0399;"],
    ["Κ", "Kappa", "&#x039A;"],
    ["Λ", "Lambda", "&#x039B;"],
    ["Μ", "Mu", "&#x039C;"],
    ["Ν", "Nu", "&#x039D;"],
    ["Ξ", "Xi", "&#x039E;"],
    ["Ο", "Omicron", "&#x039F;"],
    ["Π", "Pi", "&#x03A0;"],
    ["Ρ", "Rho", "&#x03A1;"],
    ["Σ", "Sigma", "&#x03A3;"],
    ["Τ", "Tau", "&#x03A4;"],
    ["Υ", "Upsilon", "&#x03A5;"],
    ["Φ", "Phi", "&#x03A6;"],
    ["Χ", "Chi", "&#x03A7;"],
    ["Ψ", "Psi", "&#x03A8;"],
    ["Ω", "Omega", "&#x03A9;"],
]

mathml_operators = [
    "sin",
    "cos",
    "tan",
    "sec",
    "csc",
    "cot",
    "log",
    "ln",
    "exp",
    "sqrt",
    "sum",
    "prod",
    "lim",
]


def levenshtein_similarity(var1: str, var2: str) -> float:
    """
    Compute the Levenshtein similarity between two variable names.
    The Levenshtein similarity is the ratio of the Levenshtein distance to the maximum length.
    Args:
        var1: The first variable name.
        var2: The second variable name.
    Returns:
        The Levenshtein similarity between the two variable names.
    """
    distance = Levenshtein.distance(var1, var2)
    max_length = max(len(var1), len(var2))
    similarity = 1 - (distance / max_length)
    return similarity


def jaccard_similarity(var1: str, var2: str) -> float:
    """
    Compute the Jaccard similarity between two variable names.
    The Jaccard similarity is the size of the intersection divided by the size of the union of the variable names.
    Args:
        var1: The first variable name.
        var2: The second variable name.
    Returns:
        The Jaccard similarity between the two variable names.
    """
    set1 = set(var1)
    set2 = set(var2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity


def cosine_similarity(var1: str, var2: str) -> float:
    """
    Compute the cosine similarity between two variable names.
    The cosine similarity is the dot product of the character frequency vectors divided by the product of their norms.
    Args:
        var1: The first variable name.
        var2: The second variable name.
    Returns:
        The cosine similarity between the two variable names.
    """
    char_freq1 = {char: var1.count(char) for char in var1}
    char_freq2 = {char: var2.count(char) for char in var2}

    dot_product = sum(
        char_freq1.get(char, 0) * char_freq2.get(char, 0) for char in set(var1 + var2)
    )
    norm1 = sum(freq**2 for freq in char_freq1.values()) ** 0.5
    norm2 = sum(freq**2 for freq in char_freq2.values()) ** 0.5

    similarity = dot_product / (norm1 * norm2)
    return similarity


def generate_graph(file: str = "", render: bool = False) -> pydot.Dot:
    """
    Call the REST API of math-exp-graph to convert the MathML input to its GraphViz representation
    Ensure running the REST API before calling this function
    Input: file directory or the MathML string
    Output: the GraphViz representation (pydot.Dot)
    """
    if "<math>" in file and "</math>" in file:
        content = file
    else:
        with open(file) as f:
            content = f.read()

    digraph = requests.put(
        "http://localhost:8080/mathml/math-exp-graph", data=content.encode("utf-8")
    )
    if render:
        src = Source(digraph.text)
        src.render("doctest-output/mathml_exp_tree", view=True)
    graph = pydot.graph_from_dot_data(str(digraph.text))[0]
    return graph


def generate_amatrix(graph: pydot.Dot) -> Tuple[ndarray, List[str]]:
    """
    Convert the GraphViz representation to its corresponding adjacency matrix
    Input: the GraphViz representation
    Output: the adjacency matrix and the list of the names of variables and terms appeared in the expression
    """
    node_labels = []
    for node in graph.get_nodes():
        node_labels.append(node.obj_dict["attributes"]["label"].replace('"', ""))

    amatrix = np.zeros((len(node_labels), len(node_labels)))

    for edge in graph.get_edges():
        x, y = edge.obj_dict["points"]
        label = edge.obj_dict["attributes"]["label"].replace('"', "")
        amatrix[int(x)][int(y)] = op_dict[label] if label in op_dict else 7

    return amatrix, node_labels


def heuristic_compare_variable_names(var1: str, var2: str) -> bool:
    """
    Compare two variable names in a formula, accounting for Unicode representations.
    Convert the variable names to English letter representations before comparison.

    Args:
        var1 (str): The first variable name.
        var2 (str): The second variable name.

    Returns:
        bool: True if the variable names are the same, False otherwise.
    """
    # Mapping of Greek letters to English letter representations
    greek_letters = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "θ": "theta",
        "ι": "iota",
        "κ": "kappa",
        "λ": "lambda",
        "μ": "mu",
        "ν": "nu",
        "ξ": "xi",
        "ο": "omicron",
        "π": "pi",
        "ρ": "rho",
        "σ": "sigma",
        "τ": "tau",
        "υ": "upsilon",
        "φ": "phi",
        "χ": "chi",
        "ψ": "psi",
        "ω": "omega",
    }

    # Convert Unicode representations to English letter representations
    var1 = re.sub(r"&#x(\w+);?", lambda m: chr(int(m.group(1), 16)), var1)
    var2 = re.sub(r"&#x(\w+);?", lambda m: chr(int(m.group(1), 16)), var2)

    # Convert Greek letter representations to English letter representations
    for greek_letter, english_letter in greek_letters.items():
        var1 = var1.replace(greek_letter, english_letter)
        var2 = var2.replace(greek_letter, english_letter)

    # Remove trailing quotation marks, if present
    var1 = var1.strip("'\"")
    var2 = var2.strip("'\"")

    # Compare the variable names
    return var1.lower() == var2.lower()


def extract_var_information(
    data: Dict[
        str,
        Union[
            List[
                Dict[
                    str,
                    Union[Dict[str, Union[str, int]], List[Dict[str, Union[str, int]]]],
                ]
            ],
            None,
        ],
    ]
) -> List[Dict[str, str]]:
    """
    Extracts variable information from the given JSON data of the SKEMA mention extraction.

    Parameters:
    - data (Dict[str, Union[List[Dict[str, Union[Dict[str, Union[str, int]], List[Dict[str, Union[str, int]]]]]], None]]): The input JSON data.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries containing extracted information (name, definition, and document_id).
    """
    outputs = data.get("outputs", [])
    extracted_data = []

    for output in outputs:
        attributes = output.get("data", {}).get("attributes", [])

        for attribute in attributes:
            payload = attribute.get("payload", {})
            mentions = payload.get("mentions", [])
            text_descriptions = payload.get("text_descriptions", [])

            for mention in mentions:
                name = mention.get("name", "")
                extraction_source = mention.get("extraction_source", {})
                document_reference = extraction_source.get("document_reference", {})
                document_id = document_reference.get("id", "")

                for text_description in text_descriptions:
                    description = text_description.get("description", "")
                    extraction_source = text_description.get("extraction_source", {})

                    extracted_data.append(
                        {
                            "name": name,
                            "definition": description,
                            "document_id": document_id,
                        }
                    )

    return extracted_data


def organize_into_json(
    extracted_data: List[Dict[str, str]]
) -> Dict[str, List[Dict[str, str]]]:
    """
    Organizes the extracted information into a new JSON format.

    Parameters:
    - extracted_data (List[Dict[str, str]]): A list of dictionaries containing extracted information (name, definition, and document_id).

    Returns:
    - Dict[str, List[Dict[str, str]]]: A dictionary containing organized information in a new JSON format.
    """
    organized_data = {"variables": []}

    for item in extracted_data:
        organized_data["variables"].append(
            {
                "name": item["name"],
                "definition": item["definition"],
                "document_id": item["document_id"],
            }
        )

    return organized_data


def extract_var_defs_from_metions(input_file: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Processes the input JSON file, extracts information, and writes the organized data to the output JSON file.

    Parameters:
    - input_file (str): The path to the input JSON file.

    Returns:
    - organized_data (Dict[str, List[Dict[str, str]]]): The dictionary of the variable names and their definitions.
    """
    try:
        # Read the original JSON data
        with open(input_file, "r", encoding="utf-8") as file:
            original_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: JSON decoding failed. Details: {e}")
        return {}

    # Extract information
    extracted_data = extract_var_information(original_data)

    # Organize into a new JSON format
    organized_data = organize_into_json(extracted_data)

    return organized_data


def find_definition(
    variable_name: str, extracted_data: Dict[str, List[Dict[str, str]]]
) -> str:
    """
    Finds the definition for a variable name in the extracted data.

    Args:
        variable_name (str): Variable name to find.
        extracted_data (List[Dict[str, Union[str, int]]]): List of dictionaries containing extracted information.

    Returns:
        str: Definition for the variable name, or an empty string if not found.
    """
    for attribute in extracted_data["variables"]:
        if heuristic_compare_variable_names(variable_name, attribute["name"]):
            return attribute["definition"]

    return ""


def calculate_similarity(
    definition1: str, definition2: str, field: str = "biomedical"
) -> float:
    """
    Calculates semantic similarity between two variable definitions using BERT embeddings.

    Args:
        definition1 (str): First variable definition.
        definition2 (str): Second variable definition.
        field (str): Language model to load.

    Returns:
        float: Semantic similarity score between 0 and 1.
    """
    pre_trained_model = "msmarco-distilbert-base-v2"
    model = SentenceTransformer(pre_trained_model)

    # Convert definitions to BERT embeddings
    embedding1 = model.encode(definition1, convert_to_tensor=True)
    embedding2 = model.encode(definition2, convert_to_tensor=True)

    # Calculate cosine similarity between embeddings
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()

    return cosine_similarity


def match_variable_definitions(
    list1: List[str],
    list2: List[str],
    json_path1: str,
    json_path2: str,
    threshold: float,
) -> Tuple[List[int], List[int]]:
    """
    Match variable definitions for given variable names in two lists.

    Args:
        list1 (List[str]): List of variable names from the first equation.
        list2 (List[str]): List of variable names from the second equation.
        json_path1 (str): Path to the JSON file containing variable definitions for the first article.
        json_path2 (str): Path to the JSON file containing variable definitions for the second article.
        threshold (float): Similarity threshold for considering a match.

    Returns:
        Tuple[List[int], List[int]]: Lists of indices for matched variable names in list1 and list2.
    """
    extracted_data1 = extract_var_defs_from_metions(json_path1)
    extracted_data2 = extract_var_defs_from_metions(json_path2)

    var_idx_list1 = []
    var_idx_list2 = []

    for idx1, var1 in enumerate(list1):
        max_similarity = 0.0
        matching_idx = -1
        for idx2, var2 in enumerate(list2):
            def1 = find_definition(var1, extracted_data1)
            def2 = find_definition(var2, extracted_data2)

            if def1 and def2:
                similarity = calculate_similarity(def1, def2)
                if similarity > max_similarity and similarity >= threshold:
                    max_similarity = similarity
                    matching_idx = idx2

        if matching_idx != -1:
            if idx1 not in var_idx_list1:
                var_idx_list1.append(idx1)
                var_idx_list2.append(matching_idx)

    return var_idx_list1, var_idx_list2


def get_seeds(
    node_labels1: List[str],
    node_labels2: List[str],
    method: str = "heuristic",
    threshold: float = 0.8,
    mention_json1: str = "",
    mention_json2: str = "",
) -> Tuple[List[int], List[int]]:
    """
    Calculate the seeds in the two equations.

    Args:
        node_labels1: The name lists of the variables and terms in equation 1.
        node_labels2: The name lists of the variables and terms in equation 2.
        method: The method to get seeds.
            - "heuristic": Based on variable name identification.
            - "levenshtein": Based on Levenshtein similarity of variable names.
            - "jaccard": Based on Jaccard similarity of variable names.
            - "cosine": Based on cosine similarity of variable names.
        threshold: The threshold to use for Levenshtein, Jaccard, and cosine methods.
        mention_json1: The JSON file path of the mention extraction of paper 1.
        mention_json2: The JSON file path of the mention extraction of paper 2.

    Returns:
        A tuple of two lists:
        - seed1: The seed indices from equation 1.
        - seed2: The seed indices from equation 2.
    """
    seed1 = []
    seed2 = []
    if method == "var_defs":
        seed1, seed2 = match_variable_definitions(
            node_labels1,
            node_labels2,
            json_path1=mention_json1,
            json_path2=mention_json2,
            threshold=0.9,
        )
    else:
        for i in range(0, len(node_labels1)):
            for j in range(0, len(node_labels2)):
                if method == "heuristic":
                    if heuristic_compare_variable_names(
                        node_labels1[i], node_labels2[j]
                    ):
                        if i not in seed1:
                            seed1.append(i)
                            seed2.append(j)
                elif method == "levenshtein":
                    if (
                        levenshtein_similarity(node_labels1[i], node_labels2[j])
                        > threshold
                    ):
                        if i not in seed1:
                            seed1.append(i)
                            seed2.append(j)
                elif method == "jaccard":
                    if jaccard_similarity(node_labels1[i], node_labels2[j]) > threshold:
                        if i not in seed1:
                            seed1.append(i)
                            seed2.append(j)
                elif method == "cosine":
                    if cosine_similarity(node_labels1[i], node_labels2[j]) > threshold:
                        if i not in seed1:
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


def get_union_graph(
    graph1: pydot.Dot,
    graph2: pydot.Dot,
    aligned_idx1: List[int],
    aligned_idx2: List[int],
) -> pydot.Dot:
    """
    return the union graph for visualizing the alignment results
    input: The dot representation of Graph1, the dot representation of Graph2, the aligned node indices in Graph1, the aligned node indices in Graph2
    output: dot graph
    """
    g2idx2g1idx = {str(x): str(-1) for x in range(len(graph2.get_nodes()))}
    union_graph = deepcopy(graph1)
    """
    set the aligned variables or terms as a blue circle; 
    if their names are the same, show one name; 
    if not, show two names' connection using '<<|>>'
    """
    for i in range(len(aligned_idx1)):
        if (
            union_graph.get_nodes()[aligned_idx1[i]]
            .obj_dict["attributes"]["label"]
            .replace('"', "")
            .lower()
            != graph2.get_nodes()[aligned_idx2[i]]
            .obj_dict["attributes"]["label"]
            .replace('"', "")
            .lower()
        ):
            union_graph.get_nodes()[aligned_idx1[i]].obj_dict["attributes"]["label"] = (
                union_graph.get_nodes()[aligned_idx1[i]]
                .obj_dict["attributes"]["label"]
                .replace('"', "")
                + " <<|>> "
                + graph2.get_nodes()[aligned_idx2[i]]
                .obj_dict["attributes"]["label"]
                .replace('"', "")
            )

        union_graph.get_nodes()[aligned_idx1[i]].obj_dict["attributes"][
            "color"
        ] = "blue"
        g2idx2g1idx[str(aligned_idx2[i])] = str(aligned_idx1[i])

    # represent the nodes only in graph 1 as a red circle
    for i in range(len(union_graph.get_nodes())):
        if i not in aligned_idx1:
            union_graph.get_nodes()[i].obj_dict["attributes"]["color"] = "red"

    # represent the nodes only in graph 2 as a green circle
    for i in range(len(graph2.get_nodes())):
        if i not in aligned_idx2:
            graph2.get_nodes()[i].obj_dict["attributes"]["color"] = "green"
            graph2.get_nodes()[i].obj_dict["name"] = str(len(union_graph.get_nodes()))
            union_graph.add_node(graph2.get_nodes()[i])
            g2idx2g1idx[str(i)] = str(len(union_graph.get_nodes()) - 1)

    # add the edges of graph 2 to graph 1
    for edge in union_graph.get_edges():
        edge.obj_dict["attributes"]["color"] = "red"

    for edge in graph2.get_edges():
        x, y = edge.obj_dict["points"]
        if has_edge(union_graph, g2idx2g1idx[x], g2idx2g1idx[y]):
            if (
                union_graph.get_edge(g2idx2g1idx[x], g2idx2g1idx[y])[0]
                .obj_dict["attributes"]["label"]
                .lower()
                == edge.obj_dict["attributes"]["label"].lower()
            ):
                union_graph.get_edge(g2idx2g1idx[x], g2idx2g1idx[y])[0].obj_dict[
                    "attributes"
                ]["color"] = "blue"
            else:
                e = pydot.Edge(
                    g2idx2g1idx[x],
                    g2idx2g1idx[y],
                    label=edge.obj_dict["attributes"]["label"],
                    color="green",
                )
                union_graph.add_edge(e)
        else:
            e = pydot.Edge(
                g2idx2g1idx[x],
                g2idx2g1idx[y],
                label=edge.obj_dict["attributes"]["label"],
                color="green",
            )
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


def align_mathml_eqs(
    file1: str = "",
    file2: str = "",
    mention_json1: str = "",
    mention_json2: str = "",
    mode: int = 2,
) -> Tuple[
    Any, Any, List[str], List[str], Union[int, Any], Union[int, Any], Dot, List[int]
]:
    """
    align two equation graphs using the seeded graph matching (SGD) algorithm [1].

    [1] Fishkind, D. E., Adali, S., Patsolic, H. G., Meng, L., Singh, D., Lyzinski, V., & Priebe, C. E. (2019).
    Seeded graph matching. Pattern recognition, 87, 203-215.

    Input: the paths of the two equation MathMLs; mention_json1: the mention file of paper 1; mention_json1: the mention file of paper 2;
            mode 0: without considering any priors; mode 1: having a heuristic prior
            with the similarity of node labels; mode 2: using the variable definitions
    Output:
        matching_ratio: the matching ratio between the equations 1 and the equation 2
        num_diff_edges: the number of different edges between the equations 1 and the equation 2
        node_labels1: the name list of the variables and terms in the equation 1
        node_labels2: the name list of the variables and terms in the equation 2
        aligned_indices1: the aligned indices in the name list of the equation 1
        aligned_indices2: the aligned indices in the name list of the equation 2
        union_graph: the visualization of the alignment result
        perfectly_matched_indices1: strictly matched node indices in Graph 1
    """
    graph1 = generate_graph(file1)
    graph2 = generate_graph(file2)

    amatrix1, node_labels1 = generate_amatrix(graph1)
    amatrix2, node_labels2 = generate_amatrix(graph2)

    #  If there are no mention files provided, it returns to mode 1
    if (mention_json1 == "" or mention_json2 == "") and mode == 2:
        mode = 1

    if mode == 0:
        seed1 = []
        seed2 = []
    elif mode == 1:
        seed1, seed2 = get_seeds(node_labels1, node_labels2)
    else:
        seed1, seed2 = get_seeds(
            node_labels1,
            node_labels2,
            method="var_defs",
            threshold=0.9,
            mention_json1=mention_json1,
            mention_json2=mention_json2,
        )

    partial_match = np.column_stack((seed1, seed2))

    matched_indices1, matched_indices2, _, _ = graph_match(
        amatrix1,
        amatrix2,
        partial_match=partial_match,
        padding="adopted",
        rng=rng,
        max_iter=50,
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

    small_graph_aligned = small_graph[small_graph_matched_indices][
        :, small_graph_matched_indices
    ]
    small_graph_aligned_full = np.zeros(big_graph.shape)
    small_graph_aligned_full[
        np.ix_(big_graph_matched_indices, big_graph_matched_indices)
    ] = small_graph_aligned

    num_edges = ((big_graph + small_graph_aligned_full) > 0).sum()
    diff_edges = abs(big_graph - small_graph_aligned_full)
    diff_edges[diff_edges > 0] = 1
    perfectly_matched_indices1 = check_square_array(
        diff_edges
    )  # strictly aligned node indices of Graph 1
    num_diff_edges = np.sum(diff_edges)
    matching_ratio = round(1 - (num_diff_edges / num_edges), 2)

    long_len = (
        len(node_labels1)
        if len(node_labels1) >= len(node_labels2)
        else len(node_labels2)
    )
    aligned_indices1 = np.zeros((long_len)) - 1
    aligned_indices2 = np.zeros((long_len)) - 1
    for i in range(long_len):
        if i < len(node_labels1):
            if i in matched_indices1:
                aligned_indices1[i] = matched_indices2[
                    np.where(matched_indices1 == i)[0][0]
                ]
                aligned_indices2[
                    matched_indices2[np.where(matched_indices1 == i)[0][0]]
                ] = i

    # The visualization of the alignment result.
    union_graph = get_union_graph(
        graph1,
        graph2,
        [int(i) for i in matched_indices1.tolist()],
        [int(i) for i in matched_indices2.tolist()],
    )

    return (
        matching_ratio,
        num_diff_edges,
        node_labels1,
        node_labels2,
        aligned_indices1,
        aligned_indices2,
        union_graph,
        perfectly_matched_indices1,
    )


def extract_variables_with_subsup(mathml_str: str) -> List[str]:
    # Function to extract variable names from MathML
    root = ET.fromstring(mathml_str)
    variables = []

    def process_math_element(element) -> str:
        if element.tag == "mi":  # If it's a simple variable
            variable_name = element.text
            return variable_name
        elif element.tag in ["msup", "msub", "msubsup"]:
            # Handling superscripts, subscripts, and their combinations
            base_name = process_math_element(element[0])
            if element.tag == "msup":
                modifier = "^" + process_math_element(element[1])
            elif element.tag == "msub":
                modifier = "_" + process_math_element(element[1])
            else:  # msubsup
                modifier = (
                    "_"
                    + process_math_element(element[1])
                    + "^"
                    + process_math_element(element[2])
                )
            variable_name = base_name + modifier
            return variable_name
        elif element.tag == "mrow":
            # Handling row elements by concatenating children's results
            variable_name = ""
            for child in element:
                variable_name += process_math_element(child)
            return variable_name
        elif element.tag in ["mfrac", "msqrt", "mroot"]:
            # Handling fractions, square roots, and root expressions
            base_name = process_math_element(element[0])
            if element.tag == "mfrac":
                modifier = "/" + process_math_element(element[1])
            elif element.tag == "msqrt":
                modifier = "√(" + base_name + ")"
            else:  # mroot
                modifier = "^" + process_math_element(element[1])
            variable_name = base_name + modifier
            return variable_name
        elif element.tag in ["mover", "munder", "munderover"]:
            # Handling overlines, underlines, and combinations
            base_name = process_math_element(element[0])
            if element.tag == "mover":
                modifier = "^" + process_math_element(element[1])
            elif element.tag == "munder":
                modifier = "_" + process_math_element(element[1])
            else:  # munderover
                modifier = (
                    "_"
                    + process_math_element(element[1])
                    + "^"
                    + process_math_element(element[2])
                )
            variable_name = base_name + modifier
            return variable_name
        elif element.tag in ["mo", "mn"]:
            # Handling operators and numbers
            variable_name = element.text
            return variable_name
        elif element.tag == "mtext":
            # Handling mtext
            variable_name = element.text
            return variable_name
        else:
            # Handling any other tag
            try:
                variable_name = element.text
                return variable_name
            except:
                return ""

    for elem in root.iter():
        if elem.tag in ["mi", "msup", "msub", "msubsup"]:
            variables.append(process_math_element(elem))
    result_list = list(set(variables))
    result_list = [item for item in result_list if item not in mathml_operators]
    return result_list  # Returning unique variable names


def format_subscripts_and_superscripts(latex_str: str) -> str:
    # Function to format subscripts and superscripts in a LaTeX string
    # Returns a list of unique variable names
    def replace_sub(match):
        return f"{match.group(1)}_{{{match.group(2)}}}"

    def replace_sup(match):
        superscript = match.group(2)
        return f"{match.group(1)}^{{{superscript}}}"

    pattern_sub = r"(\S+)_(\S+)"
    pattern_sup = r"(\S+)\^(\S+)"

    formatted_str = re.sub(pattern_sup, replace_sup, latex_str)
    formatted_str = re.sub(pattern_sub, replace_sub, formatted_str)

    return formatted_str


def replace_greek_with_unicode(input_str):
    # Function to replace Greek letters and their names with Unicode
    # Returns the replaced string if replacements were made, otherwise an empty string
    replaced_str = input_str
    for gl in greek_letters:
        replaced_str = replaced_str.replace(gl[0], gl[2])
        replaced_str = replaced_str.replace(gl[1], gl[2])
    return replaced_str if replaced_str != input_str else ""


def replace_unicode_with_symbol(input_str):
    # Function to replace Unicode representations with corresponding symbols
    # Returns the replaced string if replacements were made, otherwise an empty string
    pattern = r"&#x[A-Fa-f0-9]+;"
    matches = re.findall(pattern, input_str)

    replaced_str = input_str
    for match in matches:
        unicode_char = html.unescape(match)
        replaced_str = replaced_str.replace(match, unicode_char)

    return replaced_str if replaced_str != input_str else ""


def transform_variable(variable: str) -> List[Union[str, List[str]]]:
    # Function to transform a variable into a list containing different representations
    # Returns a list containing various representations of the variable
    if variable.startswith("&#x"):
        for gl in greek_letters:
            if variable in gl:
                return gl
        return [html.unescape(variable), variable]
    elif variable.isalpha():
        if len(variable) == 1:
            for gl in greek_letters:
                if variable in gl:
                    return gl
            return [variable, "&#x{:04X};".format(ord(variable))]
        else:
            return [variable]
    else:
        if len(variable) == 1:
            return [variable, "&#x{:04X};".format(ord(variable))]
        else:
            variable_list = [variable, format_subscripts_and_superscripts(variable)]
            if replace_greek_with_unicode(variable) != "":
                variable_list.append(replace_greek_with_unicode(variable))
                variable_list.append(
                    replace_greek_with_unicode(
                        format_subscripts_and_superscripts(variable)
                    )
                )
            if replace_unicode_with_symbol(variable) != "":
                variable_list.append(replace_unicode_with_symbol(variable))
                variable_list.append(format_subscripts_and_superscripts(variable))

            return variable_list


def create_variable_dictionary(
    variables: List[str],
) -> Dict[str, List[Union[str, List[str]]]]:
    # Function to create a dictionary mapping variables to their representations
    # Returns a dictionary with variables as keys and their representations as values
    variable_dict = {}
    for variable in variables:
        variable_dict[variable] = transform_variable(variable)
    return variable_dict


def generate_variable_dict(mathml_string):
    # Function to generate a variable dictionary from MathML
    try:
        variables = extract_variables_with_subsup(mathml_string)
        variable_dict = create_variable_dictionary(variables)
        return variable_dict
    except:
        return {}
