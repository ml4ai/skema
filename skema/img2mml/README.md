# img2mml

This directory contains the code for the img2mml service, which processes images
of equations and returns presentation MathML corresponding to those equations.

The model was developed by Gaurav Sharma, Clay Morrison and Liang Zhang, and this wrapper
service was developed by Deepsana Shahi, Adarsh Pyarelal and Liang Zhang.

The model itself is not checked into the repository, but you can get it from
here:
https://artifacts.askem.lum.ai/skema/img2mml/models/cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt

Place the model file in the `trained_models` directory.

The curl command below should do the trick.

```
curl -L https://artifacts.askem.lum.ai/skema/img2mml/models/cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt > trained_models/cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt
```

If you have the checkpoint in the `trained_models` directory already and hope to update it, please run the above curl command that will replace the previous one.

To update the model name or path, please make the following modifications to support updating the img2mml service and the corresponding Docker operations:

1. Modify the paths of the model_path variable in the get_mathml_from_bytes function in the skema/img2mml/api.py file.
2. Update the path settings in the "retrieve latest model for img2mml component" section of skema/.github/workflows/docs.yml.
3. Adjust the curl command in the test_equation_reading section of skema/.drone.yml to download the checkpoint.
4. Update the download checkpoint path in skema/img2mml/README.md.

These changes will ensure that the necessary files and paths are updated correctly.

Then, run the invocation below to launch the Dockerized service:

```
docker-compose up --build
```

To test the service without Docker (e.g., for development purposes), ensure
that you have installed the required packages (run `pip install -e .[img2mml]`
in the root of the repository).

Then, run the following command to launch the img2mml server program:

```
uvicorn skema.img2mml.img2mml:app --reload
```

An example test program is provided as well, which you can invoke with:
Make sure that you are in generate_mathml folder.

```
python img2mml_demo.py
```
