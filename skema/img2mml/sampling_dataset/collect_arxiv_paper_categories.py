import json
import os
import re
import time
from typing import Dict, List

import requests


def get_paper_categories(arxiv_ids: List[str]) -> Dict[str, List[str]]:
    """
    Get the categories for a list of arXiv IDs. The arXiv API only allows a maximum of 10 IDs per request.

    Args:
        arxiv_ids (List[str]): A list of arXiv IDs. The IDs should be in the format "YYMM.NNNNN".

    Returns:
        Dict[str, List[str]]: A dictionary mapping arXiv IDs to a list of categories. The categories are strings. If an ID is not found, it will not be included in the dictionary.
    """
    base_url = "http://export.arxiv.org/api/query?id_list="
    url = base_url + ",".join(arxiv_ids)

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes.

        # Parse the XML response using regex.
        xml_data = response.text

        # Use regex to remove the first <?xml> declaration.
        xml_data = re.sub(r"<\?xml[^>]*>", "", xml_data, count=1)

        categories_dict = {}

        # Use regex to find all entry blocks in the XML.
        entry_blocks = re.findall(r"<entry>(.*?)</entry>", xml_data, re.DOTALL)

        for entry_block in entry_blocks:
            # Use regex to extract the arXiv ID.
            arxiv_id_match = re.search(r"<id>(.*?)</id>", entry_block)
            if arxiv_id_match:
                arxiv_id = arxiv_id_match.group(1)
                arxiv_id = arxiv_id.replace("http://arxiv.org/abs/", "")  # remove URL
                arxiv_id = arxiv_id[:-2]  # remove version number

                # Use regex to find all category elements for this entry.
                category_matches = re.findall(r'<category term="([^"]+)"', entry_block)

                categories = list(set(category_matches))
                categories_dict[arxiv_id] = categories

        return categories_dict

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        raise e


def load_json(file_path: str) -> List[str]:
    """
    Load arXiv IDs from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing arXiv IDs.

    Returns:
        List[str]: A list of arXiv IDs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. This file is required.")
    with open(file_path) as f:
        arxiv_ids: List[str] = json.load(f)
    return arxiv_ids


def save_arxiv_categories(file_path: str, data: Dict[str, List[str]]):
    """
    Save arXiv paper categories to a JSON file.

    Args:
        file_path (str): Path to the JSON file where categories will be saved.
        data (Dict[str, List[str]]): A dictionary mapping arXiv IDs to categories.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    start_time = time.perf_counter()
    script_dir = os.path.dirname(__file__)

    arxiv_ids_file = os.path.join(script_dir, "paper_data/arxiv_paper_ids.json")
    final_categories_file = os.path.join(
        script_dir, "paper_data/arxiv_paper_categories.json"
    )

    arxiv_ids = load_json(arxiv_ids_file)
    arxiv_ids = list(set(arxiv_ids))

    try:
        final_categories = load_json(final_categories_file)
    except FileNotFoundError:
        final_categories = {}
        print("No existing categories file found. Starting from scratch.")

    batch_size = 10  # This is the maximum number of IDs that can be queried at once.

    for i, start_idx in enumerate(range(0, len(arxiv_ids), batch_size)):
        print(f"Processing batch {i + 1} of {len(arxiv_ids) // batch_size + 1}")
        ids = arxiv_ids[start_idx : start_idx + batch_size]
        ids = [
            id for id in ids if id not in final_categories
        ]  # Only get categories for papers that we don't already have

        if len(ids) == 0:
            continue

        categories = {}
        try:
            categories = get_paper_categories(ids)
        except:
            print(
                "Error getting categories. If this is a timeout error, you may need to wait a few minutes and try again."
            )
            retry = (
                input(
                    "Enter 'y' to wait 15 minutes and retry, or 'n' to skip this batch. If you choose to skip, you can always rerun the script later and it will pick up where it left off."
                ).lower()[0]
                == "y"
            )
            if retry:
                time.sleep(900)
                categories = get_paper_categories(ids)
            else:
                continue
        final_categories.update(categories)

        if i > 0 and i % 10 == 0:
            # Sleep for 5 seconds every 10 requests
            print("Sleeping for 5 seconds...")
            time.sleep(5)

    print(f"Number of papers processed: {len(final_categories)}")

    # Write results to a file
    save_arxiv_categories(final_categories_file, final_categories)

    end_time = time.perf_counter()
    print("Done! Results saved to arxiv_paper_categories.json")
    print(f"Time elapsed: {end_time - start_time} seconds.")


if __name__ == "__main__":
    main()
