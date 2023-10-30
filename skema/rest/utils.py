from collections import defaultdict
from typing import Any, Dict
import itertools as it
from askem_extractions.data_model import AttributeCollection, AttributeType, Mention
from bs4 import BeautifulSoup, Comment

from skema.rest.schema import TextReadingEvaluationResults, AMRLinkingEvaluationResults


def fn_preprocessor(fn_data):

    logs = []

    '''
    We will currently preprocess based on 2 different common bugs
    1) wire tgt's being -1 -> which we will delete these wires
    2) metadata being inline for bf entries instead of an index into the metadata_collection -> which we will replace with an index of 2
    3) missing function_type field on a bf entry -> will replace with function_type: "OTHER"
    4) If there is not a body field to a function -> replace "FUNCTION" with "ABSTRACT and set "name":"unknown"
    5) NOT DONE YET: In the future we will preprocess about function calls being arguments, in order to simplify extracting the dataflow 
    '''

    # first we check the top bf level of wires and inline metadata: 
    keys_to_check = ['bf', 'wff', 'wfopi', 'wfopo', 'wopio']
    for key in keys_to_check:
        if key == 'bf':
            try:
                for (i, entry) in enumerate(fn_data['modules'][0]['fn'][key]):
                    try: 
                        metadata_obj = entry['metadata']
                        if not isinstance(metadata_obj, int):
                            entry['metadata'] = 2
                            logs.append(f"Inline metadata on {i+1}'th entry in top level bf")
                    except:
                        continue
                    try: 
                        temp = entry['function_type']
                    except:
                        entry['function_type'] = "IMPORTED"
                        logs.append(f"Missing function_type on {i+1}'th entry in top level bf")
                    try: 
                        if entry['function_type'] == "FUNCTION":
                            temp = entry['body']
                    except:
                        entry['function_type'] = "ABSTRACT"
                        entry['name'] = "Unknown"
                        logs.append(f"Missing Function body on {i+1}'th entry in top level bf")    
            except:
                continue
        else:
            try: 
                for (i, entry) in enumerate(fn_data['modules'][0]['fn'][key]):
                    if entry['tgt'] == -1:
                        del fn_data['modules'][0]['fn'][key][i]
                        logs.append(f"The {i+1}'th {key} wire in the top level bf is targeting -1")
            except:
                continue

    # now we iterate through the fn_array and do the same thing
    for (j,fn_ent) in enumerate(fn_data['modules'][0]['fn_array']):
        for key in keys_to_check:
            if key == 'bf':
                try:
                    for (i, entry) in enumerate(fn_ent[key]):
                        try: 
                            metadata_obj = entry['metadata']
                            if not isinstance(metadata_obj, int):
                                entry['metadata'] = 2
                                logs.append(f"Inline metadata on {i+1}'th bf in the {j+1}'th fn_array")
                        except:
                            continue
                        try: 
                            temp = entry['function_type']
                        except:
                            entry['function_type'] = "IMPORTED"
                            logs.append(f"Missing function_type on {i+1}'th bf in the {j+1}'th fn_array")
                        try: 
                            if entry['function_type'] == "FUNCTION":
                                temp = entry['body']
                        except:
                            entry['function_type'] = "ABSTRACT"
                            entry['name'] = "Unknown"
                            logs.append(f"Missing Function body on {i+1}'th bf in the {j+1}'th fn_array")  
                except:
                    continue
            else:
                try: 
                    for (i, entry) in enumerate(fn_ent[key]):
                        if entry['tgt'] == -1:
                            del fn_ent[key][i]
                            logs.append(f"The {i+1}'th {key} wire in the {j+1}'th fn_array is targeting -1")
                except:
                    continue

    return fn_data, logs

def clean_mml(mml: str) -> str:
    """Cleans/sterilizes pMML for AMR generation service"""
    # FIXME: revisit if JSON deserialization on MORAE side changes
    to_remove = ["alttext", "display", "xmlns", "mathvariant", "class"]
    soup = BeautifulSoup(mml, "html.parser")
    # remove comments
    for comment in soup(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # prune attributes
    for attr in to_remove:
        for tag in soup.find_all(attrs={attr: True}):
            del tag[attr]
    return str(soup).replace("\n", "")


def extraction_matches_annotation(mention: Mention, annotation: Dict[str, Any]) -> bool:
    """ Determines whether the extraction matches the annotation"""

    # First iteration of the matching algorithm

    # Get the annotation's text
    gt_text = annotation["text"]

    # Get the extractions text
    m_text = mention.extraction_source.surrounding_passage

    return gt_text in m_text


def compute_text_reading_evaluation(gt_data: list, attributes: AttributeCollection) -> TextReadingEvaluationResults:
    """ Compute the coverage of text reading extractions """

    # Get the extractions from the attribute collection
    extractions = [a.payload for a in attributes.attributes if a.type == AttributeType.anchored_entity]

    # Get the extraction annotations from the ground truth data
    annotations_by_page = defaultdict(list)
    for a in gt_data:
        if a["type"] == "Highlight" and a["color"] == "#f9cd59":
            page = a["page"]
            annotations_by_page[page].append(a)

    # Count the matches
    num_matches = 0
    for e in extractions:
        for m in e.mentions:
            if m.extraction_source is not None:
                te = m.extraction_source
                if te.page is not None:
                    e_page = te.page
                    page_annotations = annotations_by_page[e_page]
                    for a in page_annotations:
                        if extraction_matches_annotation(m, a):
                            num_matches += 1
                            break

    return TextReadingEvaluationResults(
        num_manual_annotations=len(gt_data),
        yield_=len(extractions),
        correct_extractions=num_matches,
        precision=num_matches / len(gt_data)
    )


greek_alphabet = {
    'Α': 'alpha',
    'α': 'alpha',
    'Β': 'beta',
    'β': 'beta',
    'Γ': 'gamma',
    'γ': 'gamma',
    'Δ': 'delta',
    'δ': 'delta',
    'Ε': 'epsilon',
    'ε': 'epsilon',
    'Ζ': 'zeta',
    'ζ': 'zeta',
    'Η': 'eta',
    'η': 'eta',
    'Θ': 'theta',
    'θ': 'theta',
    'Ι': 'iota',
    'ι': 'iota',
    'Κ': 'kappa',
    'κ': 'kappa',
    'Λ': 'lambda',
    'λ': 'lambda',
    'Μ': 'mu',
    'μ': 'mu',
    'Ν': 'nu',
    'ν': 'nu',
    'Ξ': 'xi',
    'ξ': 'xi',
    'Ο': 'omicron',
    'ο': 'omicron',
    'Π': 'pi',
    'π': 'pi',
    'Ρ': 'rho',
    'ρ': 'rho',
    'Σ': 'sigma',
    'σ': 'sigma',
    'ς': 'sigma',
    'Τ': 'tau',
    'τ': 'tau',
    'Υ': 'upsilon',
    'υ': 'upsilon',
    'Φ': 'phi',
    'φ': 'phi',
    'Χ': 'chi',
    'χ': 'chi',
    'Ψ': 'psi',
    'ψ': 'psi',
    'Ω': 'omega',
    'ω': 'omega'
}

def compute_amr_linking_evaluation(linked_amr, gt_linked_amr) -> AMRLinkingEvaluationResults:

    # Find the amr elements with metadata in the GT
    gt_amr_ids = {m['amr_element_id'] for m in gt_linked_amr['metadata'] if m['amr_element_id'] is not None}

    # Fetch the relevant elements from both amrs
    def get_elem_by_id(data, ids):
        ret = list()
        if isinstance(data, list):
            ret.extend(it.chain.from_iterable(get_elem_by_id(a, ids) for a in data))
        elif isinstance(data, dict):
            if "id" in data and data["id"] in ids:
                ret.append(data)
            else:
                ret.extend(it.chain.from_iterable(get_elem_by_id(v, ids) for k, v in data.items() if k != "metadata"))
        return ret

    gt_elems = get_elem_by_id(gt_linked_amr, gt_amr_ids)
    runtime_elems = get_elem_by_id(linked_amr, gt_amr_ids)

    # Generate metadata dictionaries
    gt_metadata = defaultdict(list)
    for m in gt_linked_amr['metadata']:
        gt_metadata[m['amr_element_id']].append(m)

    runtime_metadata = defaultdict(list)
    for m in linked_amr['metadata']['attributes']:
        runtime_metadata[m['amr_element_id']].append(m)

    # Compute the numbers
    tp, tn, fp, fn = 0, 0, 0, 0

    for amr_id in gt_amr_ids:
        gt = gt_metadata[amr_id]
        rt = runtime_metadata[amr_id]

        # Get the text from the ground truth
        gt_texts = {e['text'] for e in gt}
        expanded_gt_texts = set()
        for t in gt_texts:
            for k, v in greek_alphabet.items():
                if k in t:
                    expanded_gt_texts.add(t.replace(k, v))
        gt_texts |= expanded_gt_texts

        # Get the text from the automated extractions
        rt_texts = set()
        for e in rt:
            e = e['payload']
            for m in e['mentions']:
                name = m['name']
                for d in e['text_descriptions']:
                    desc = d['description']
                    rt_texts.add((name, desc))
                for v in e['value_descriptions']:
                    val = v['value']['amount']
                    rt_texts.add((name, val))

        # Compute hits and misses
        if len(gt_texts) > 0:
            hit = False
            for gtt in gt_texts:
                if not hit:
                    for (a, b) in rt_texts:
                        # Both the name and the desc have to be present in the
                        # annotation in order to be a "hit"
                        if a in gtt and b in gtt:
                            tp += 1
                            hit = True
                            break
            # If we made it to this point and neither of the extractions matched
            # then, this is a false negative
            fn += 1
        elif len(rt_texts) > 0:
            fp += 1
        else:
            tn += 1

    precision = tp / ((tp + fp) + 0.000000001)
    recall = tp / ((tp + fn) + 0.000000001)

    f1 = (2*precision*recall) / ((precision + recall)  + 0.000000001)

    return AMRLinkingEvaluationResults(
        num_gt_elems_with_metadata=len(gt_amr_ids),
        precision=precision,
        recall=recall,
        f1=f1
    )
