from collections import defaultdict
from typing import Any, Dict, List, Tuple

from askem_extractions.data_model import AttributeCollection, AttributeType, AnchoredEntity, Mention
from bs4 import BeautifulSoup, Comment

from skema.rest.schema import TextReadingEvaluationResults


def fn_preprocessor(fn_data: Dict) -> Tuple[Dict, List]:

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
                        None
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
                None
        else:
            try: 
                for (i, entry) in enumerate(fn_data['modules'][0]['fn'][key]):
                    if entry['tgt'] == -1:
                        del fn_data['modules'][0]['fn'][key][i]
                        logs.append(f"The {i+1}'th {key} wire in the top level bf is targeting -1")
            except:
                None

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
                            None
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
                    None
            else:
                try: 
                    for (i, entry) in enumerate(fn_ent[key]):
                        if entry['tgt'] == -1:
                            del fn_ent[key][i]
                            logs.append(f"The {i+1}'th {key} wire in the {j+1}'th fn_array is targeting -1")
                except:
                    None

    print(logs)
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
