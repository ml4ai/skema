import json

import requests


def get_paper_category(arxiv_id):
    # Define the base URL of the arXiv API.
    base_url = "http://export.arxiv.org/api/query?id_list="

    # Construct the full URL with the paper ID.
    url = base_url + arxiv_id

    try:
        # Send a GET request to the arXiv API.
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes.

        # Parse the XML response.
        xml_data = response.text

        # arXiv API returns data in XML format, so we need to parse it.
        # You can use an XML parsing library, but for simplicity, let's use regex.
        import re

        # Use regex to find the category element.
        category_match = re.search(r'<category term="(.*?)"/>', xml_data)

        if category_match:
            category = category_match.group(1)
            return category
        else:
            return "Category not found"

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    arxiv_id = "1501"
    category = get_paper_category(arxiv_id)
    if category:
        print(f"The paper is in the category: {category}")
    else:
        print("Failed to retrieve the category information.")
