import json
import os
from typing import Dict, List

from PIL import Image
from pix2tex.cli import LatexOCR

# Define constants
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
BASE_IMAGE_PATH = os.path.join(SCRIPT_DIRECTORY, "./img2latex_100k/formula_images")


# Load lines from a file and remove trailing whitespace
def load_file_lines(file_path: str) -> List[str]:
    with open(file_path, "r", newline="\n") as f:
        return [line.rstrip() for line in f]


# Save data to a JSON file with indentation
def save_data(data: Dict, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def main() -> None:
    # Initialize the LatexOCR model
    model = LatexOCR()

    # Load test data and formula list
    test_data = load_file_lines(
        os.path.join(SCRIPT_DIRECTORY, "./img2latex_100k/im2latex_test.lst")
    )
    formula_list = load_file_lines(
        os.path.join(SCRIPT_DIRECTORY, "./img2latex_100k/im2latex_formulas.norm.lst")
    )

    results = {}  # Using a dictionary to store results
    errors = []

    # Process each test data entry
    for i, data in enumerate(test_data):
        formula_idx, image_id, _ = data.split()
        image_path = os.path.join(BASE_IMAGE_PATH, f"{image_id}.png")

        print(f"{i+1} Processing image {image_id}", end=" | ")
        img = Image.open(image_path)

        try:
            predicted_latex = model(img)
        except:
            print(f"Failed to predict", end=" | ")
            predicted_latex = "Error"
            errors.append(image_id)

        print(f"Done")

        # Store result along with ground truth using image ID as key
        results[image_id] = {
            "latex": predicted_latex,
            "gt_latex": formula_list[int(formula_idx)],
            "line_number": formula_idx,
        }

        # Save results and errors in batches of 500
        if (i + 1) % 500 == 0:
            print(f"\n---\nSaving current {i+1} results to files\n---\n")
            save_data(
                results,
                os.path.join(SCRIPT_DIRECTORY, "img2latex_100k_test_results.json"),
            )
            save_data(
                errors,
                os.path.join(SCRIPT_DIRECTORY, "img2latex_100k_test_errors.json"),
            )

    # Save final results and errors
    save_data(
        results, os.path.join(SCRIPT_DIRECTORY, "img2latex_100k_test_results.json")
    )
    save_data(errors, os.path.join(SCRIPT_DIRECTORY, "img2latex_100k_test_errors.json"))
    print("Done")


if __name__ == "__main__":
    main()
