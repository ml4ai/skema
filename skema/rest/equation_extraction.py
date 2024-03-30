import os
import requests
import json
from time import sleep
from IPython.display import clear_output


def process_images_in_folder(folder_path: str, gpt_key: str) -> None:
    """
    Process PNG images in a folder to detect equations using an API.

    Args:
        folder_path (str): Path to the folder containing PNG images.
        gpt_key (str): API key for accessing the equation detection service.

    Returns:
        None
    """
    # URL for equation detection service
    url = "http://54.227.237.7/integration/equation_classifier"

    # Ensure the API key is available
    if not gpt_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Append the API key to the URL as a query parameter
    url_with_key = f"{url}?gpt_key={gpt_key}"

    # Dictionary to store results
    results = []

    # Iterate over PNG files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            files = {"image": (filename, open(image_path, "rb"), "image/png")}

            # Send POST request to the equation detection service
            response = requests.post(url_with_key, files=files)

            # Close the file
            files["image"][1].close()

            # Check response status code
            if response.status_code == 200:
                data = response.json()
                result = {
                    "filename": filename,
                    "contains_equation": data["is_equation"],
                    "latex_equation": data["equation_text"],
                }
                results.append(result)
            else:
                # If request fails, add default result and raise an error
                result = {
                    "filename": filename,
                    "contains_equation": False,
                    "latex_equation": None,
                }
                results.append(result)
                print(
                    f"Request for {filename} failed with status code:",
                    response.status_code,
                )

            # Sleep to avoid overwhelming the API
            sleep(3)

    # Write results to a JSON file
    output_file = f"{folder_path}/equation_results.json"
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Results written to", output_file)


COSMOS_BASE_URL: str = "http://cosmos0004.chtc.wisc.edu:8088/cosmos_service"


def download_images_from_pdf(pdf_local_path: str, save_folder: str) -> None:
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Submit the locally copied PDF to the COSMOS processing pipeline
    submit_endpoint: str = COSMOS_BASE_URL + "/process/"
    with open(pdf_local_path, "rb") as pdf_to_parse:
        file_form: dict = {"pdf": pdf_to_parse}
        data_form: dict = {"compress_images": False}
        response: requests.Response = requests.post(
            submit_endpoint, files=file_form, data=data_form
        )

        response_data: dict = response.json()
        job_id: str = response_data["job_id"]

        status_endpoint: str = response_data["status_endpoint"]
        results_endpoint: str = response_data["result_endpoint"]

    POLL_COUNT: int = 80
    POLL_INTERVAL: int = 5

    job_done: bool = False

    for i in range(POLL_COUNT):
        response: requests.Response = requests.get(status_endpoint)
        response_data: dict = response.json()
        clear_output(wait=True)
        print(f"Polled status endpoint {i} times:\n{response_data}")
        job_done: bool = response_data["error"] or response_data["job_completed"]
        if job_done:
            break
        sleep(POLL_INTERVAL)

    if not job_done:
        raise RuntimeError(
            f"Job not complete after {POLL_COUNT * POLL_INTERVAL} seconds."
        )
    elif response_data["error"]:
        raise RuntimeError(f"An unexpected error occurred: {response_data['error']}")
    else:
        print(
            f"Job succeeded after {response_data['time_processing']} seconds.\n"
            f"Results can be viewed at {results_endpoint}"
        )

    # Extracted document equations, bounding boxes, and images
    equation_data: dict = requests.get(
        f"{results_endpoint}/extractions/equations"
    ).json()

    # Download images
    for equation in equation_data:
        img_url: str = equation["img_pth"]
        img_name: str = img_url.split("/")[-1]
        img_save_path: str = os.path.join(save_folder, img_name)
        try:
            img_response: requests.Response = requests.get(img_url)
            with open(img_save_path, "wb") as img_file:
                img_file.write(img_response.content)
            print(f"Image downloaded: {img_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to download image {img_name}: {e}")


def process_pdf_and_images(pdf_local_path: str, save_folder: str, gpt_key: str) -> None:
    """
    Download images from a PDF file and then process them to detect equations.

    Args:
        pdf_local_path (str): Path to the local PDF file.
        save_folder (str): Path to the folder where images will be saved.
        gpt_key (str): API key for accessing the equation detection service.

    Returns:
        None
    """
    # Download images from the PDF file
    download_images_from_pdf(pdf_local_path, save_folder)

    # Process the downloaded images to detect equations
    process_images_in_folder(save_folder, gpt_key)
