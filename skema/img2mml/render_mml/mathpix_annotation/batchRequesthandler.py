"""
This program sends requests to the Mathpix server.

Original: https://github.com/imzoc/mathpix-annotation/tree/master
Modified by Adi
"""


import os
import requests
import config
import json


class BatchRequestHandler:
    def __init__(self):
        api_key = config.MATHPIX_API_KEY
        self.url = "https://api.mathpix.com/v3/batch"

        self.headers = {
            "app_id": "Zach's MathML requests",
            "app_key": api_key,
        }

        with open("../image_ids.json", "r") as f:
            image_ids = json.load(f)[0:5]  # HARD limitnig to two images for now
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

    def post_request(self):
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

    def get_request(self, batch_id):
        print(f"Attempting Mathpix GET request for batch {batch_id}")
        request = requests.get(
            os.path.join(self.url, str(batch_id)),
            headers=self.headers,
        )
        reply_json = request.json()
        print("Request Complete")
        return reply_json
