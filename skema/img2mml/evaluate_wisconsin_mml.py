import os
import json
import requests
from typing import List, Dict
from torchtext.data.metrics import bleu_score
from preprocessing.preprocess_mml import simplification
from time import sleep
import re


def remove_mathml_attributes(mathml: str) -> str:
    """
    Remove attributes from MathML elements.

    Args:
        mathml (str): The input MathML expression.

    Returns:
        str: The MathML expression with attributes removed.
    """
    # Remove attributes from MathML elements
    mathml = re.sub(r"<\s*(\w+)\s+[^>]*>", r"<\1>", mathml)
    return mathml


def normalize_mathml(mathml: str) -> str:
    """
    Normalize the given MathML expression.

    Args:
        mathml (str): The input MathML expression.

    Returns:
        str: The normalized MathML expression.
    """
    # Remove attributes from <mrow> wrapped <mi> elements with mathvariant attribute
    mathml = re.sub(
        r'<mrow>\s*<mi\s+mathvariant="[^"]*">([^<]*)</mi>\s*</mrow>',
        r"<mi>\1</mi>",
        mathml,
    )

    # Remove attributes from all MathML elements
    mathml = remove_mathml_attributes(mathml)

    return mathml


def remove_mrow_elements(mathml: str) -> str:
    """
    Remove all <mrow> and </mrow> elements from the MathML expression.

    Args:
        mathml (str): The input MathML expression.

    Returns:
        str: The MathML expression with <mrow> and </mrow> elements removed.
    """
    # Remove <mrow> and </mrow> elements
    mathml = re.sub(r"<mrow>|</mrow>", "", mathml)
    return mathml


def simplify_mathml(mathml: str) -> str:
    """
    Simplify a given mathml expression.

    Args:
        mathml (str): The input mathml expression.

    Returns:
        str: The simplified mathml expression.
    """
    # Remove newline characters
    cleaned_mathml = str(mathml).replace("\n", "")

    # Simplify the math expression
    try:
        simplified_mathml = simplification(cleaned_mathml)
    except:
        simplified_mathml = ""

    return simplified_mathml


def process_wisconsin_mml(
    wisconsin_mml_path: str,
    images_folder_path: str,
    output_file_path: str,
    remove_mrow: bool = True,
) -> Dict[str, float]:
    """
    Process Wisconsin MML data, call API, simplify mathml, and calculate BLEU scores.

    Args:
        wisconsin_mml_path (str): Path to the Wisconsin MML JSON file.
        images_folder_path (str): Path to the folder containing equation images.
        output_file_path (str): Path to the output text file.
        remove_mrow (bool): If removing all <mrow> and </mrow> before calculating BLEU score.

    Returns:
        Dict[str, float]: A dictionary with mathml ID as key and average BLEU score as value.
    """
    SERVICE_ADDRESS = os.environ.get(
        "SKEMA_ADDRESS", "https://api.askem.lum.ai/eqn2mml/image/mml"
    )
    bleu_scores: Dict[str, List[float]] = {}

    with open(wisconsin_mml_path, "r") as json_file:
        wisconsin_mml_data: Dict[str, str] = json.load(json_file)

    for math_id, simplified_mathml in wisconsin_mml_data.items():
        filename = os.path.join(images_folder_path, math_id + ".png")
        if os.path.exists(filename):
            files = {"data": open(filename, "rb")}
            r = requests.post(SERVICE_ADDRESS, files=files)
            api_mathml = r.text.strip()
            while "503 Service Temporarily Unavailable" in api_mathml:
                r = requests.post(SERVICE_ADDRESS, files=files)
                api_mathml = r.text.strip()
                sleep(180)

            sleep(10)
            if api_mathml:
                prediction = normalize_mathml(simplify_mathml(api_mathml))
                if prediction == "":
                    continue
                ground_truth = normalize_mathml(simplified_mathml)
                if remove_mrow:
                    prediction = remove_mrow_elements(prediction)
                    ground_truth = remove_mrow_elements(ground_truth)

                bleu = bleu_score(
                    [prediction.split()],
                    [[ground_truth.split()]],
                )
                if bleu > 0:
                    bleu_scores[math_id] = bleu
                    print(math_id + ": " + str(bleu))

    with open(output_file_path, "w") as output_file:
        for math_id, bleu in bleu_scores.items():
            output_file.write(f"{math_id}: {bleu}\n")

    avg_bleu = {
        str(math_id): str(sum(scores) / len(scores)) for math_id, scores in bleu_scores.items()
    }
    return avg_bleu


# Specify input JSON file path, image folder path, and output text file path
wisconsin_mml_path: str = "wisconsin_mml.json"
images_folder_path: str = "equation-images/images"
output_file_path: str = "bleu_scores.txt"

# Call the function to process Wisconsin MML data, call API, simplify mathml, calculate BLEU scores,
# and write BLEU scores to the output text file
avg_bleu_scores = process_wisconsin_mml(
    wisconsin_mml_path, images_folder_path, output_file_path
)

# Calculate mean and standard deviation of BLEU scores
bleu_values = list(avg_bleu_scores.values())
mean_bleu = sum(bleu_values) / len(bleu_values)
std_dev_bleu = (
    sum((x - mean_bleu) ** 2 for x in bleu_values) / len(bleu_values)
) ** 0.5

print("Mean BLEU Score:", mean_bleu)
print("Standard Deviation of BLEU Scores:", std_dev_bleu)
