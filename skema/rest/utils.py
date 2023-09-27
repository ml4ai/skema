from collections import defaultdict
from typing import Any, Dict

from askem_extractions.data_model import AttributeCollection, AttributeType, AnchoredEntity, Mention
from bs4 import BeautifulSoup, Comment

from skema.rest.schema import TextReadingEvaluationResults


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
