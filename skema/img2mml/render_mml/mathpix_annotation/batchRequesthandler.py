"""
This program sends requests to the Mathpix server. This handler is only for images stored
in the original git repo linked below. The image_ids.json file is used to get the specific images
from the raws on github. There is no way to specify the source image URLs outside the handler.

Original: https://github.com/imzoc/mathpix-annotation/tree/master
Modified by Adi
"""


import os
import requests
import config
import json


class BatchRequestHandler:
    def __init__(self):
        self.url = "https://api.mathpix.com/v3/batch"

        self.headers = {
            "app_id": "Zach's MathML requests",
            "app_key": config.MATHPIX_API_KEY,
        }

        with open("../image_ids.json", "r") as f:
            image_ids = json.load(f)

        self.json = {
            "urls": {
                image_id: f"https://raw.githubusercontent.com/imzoc/mathpix-annotation/master/mathml-images/images_filtered/{image_id}.png"
                for image_id in image_ids
            },
            "ocr_behavior": "text",
            "formats": ["data", "html", "text"],
            "data_options": {
                "include_mathml": True,
                "include_latex": True,
            },
        }

    def post_batch(self):
        """
        Sends a batch request to the Mathpix API containing the images in image_ids.json.

        API Reference: https://docs.mathpix.com/#process-a-batch

        Raises:
            RuntimeError: In case there in an error with the API request.

        Returns:
            int: The batch id for the request. Use this id later to get the results form the API.
        """
        print("Attempting Mathpix POST request")
        request = requests.post(
            self.url,
            json=self.json,
            headers=self.headers,
            timeout=30,
        )

        reply_json = request.json()
        if "batch_id" not in reply_json:
            raise RuntimeError(
                "Batch ID not returned. Check the retruned error message.", reply_json
            )

        print("Request Complete")

        return reply_json["batch_id"]

    def get_batch_results(self, batch_id):
        """
        Gets the results of the batch with the given batch ID. 5 images
        take approximately 1 secoond to process, so results may be incomplete until
        enough time has passed.

        API Reference: https://docs.mathpix.com/#process-a-batch

        Args:
            batch_id (int): The batch ID returned after posting the batch request.

        Returns:
            dict: A JSON containing the results.
        """
        print(f"Attempting Mathpix GET request for batch {batch_id}")
        request = requests.get(
            os.path.join(self.url, str(batch_id)),
            headers=self.headers,
        )

        reply_json = request.json()

        print("Request Complete")

        return reply_json
