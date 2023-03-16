[#](#) img2mml

This directory contains the code for the img2mml service, which processes images
of equations and returns presentation MathML corresponding to those equations.

The model was developed by Gaurav Sharma and Clay Morrison, and this wrapper
service was developed by Deepsana Shahi and Adarsh Pyarelal.

## Requirements
```
python3 -m pip install -r requirements.txt
```

To install Pytorch
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

To install Torchtext:
```
pip install torchtext==0.6.0
```
## Generating raw dataset
Please check out the `data_generation/README.md` file.

## Creating dataset for training
We need to create dataset for training purpose from the raw dataset that has been generated from the arxiv source files. For further details about data generation, check out `data_generation/README.md`.

#### Distribution
We need to provide the distribution which we want to follow while sampling our dataset for training purpose. To do this, update the `sampling_dataset/sampling_config.json`.

The dataset distribution is largely divided into a few bin based on the length of the MathML equations. These bins are:
[0,50], [50,100], [100,150], [150,200], [200,250], [250,300], [300,350], [350,1000]. In every bin represents range of the number of tokens a MathML equation can have.

To sample dataset from the year(s) i.e. considering all months in that year(s), set `sample_from_years` to true in the sampling_config file. To sample from specific month folders, set `sample_from_years` to false and `sample_from_months` to true.

The `src_path` will be the destination path of the `data_generation/data_generation_config.json`.

After updating the `sampling_config.json` file,
```
python create_dataset.py
```

It will create sample data at `training_data/sample_data` as per desired distribution. Under sample_data folder, there should be following itmes: an "original_mml.lst" and "original_latex.lst" file containing all the MathMLs and LaTeXs, a "paths.lst" file containing all the paths from where the MathML are fetched, and an "images" folder. The paths defined in "paths.lst" can be seen in a specific way i.e. "<yr> <month> <folder> <type_of_eqn> <eqn_num>". This gives all the information we need to grab the specific file.

## Training

Before starting the training, please ensure that the dataset is at `skema/skema/img2mml/training_data` folder. The `training_data` folder will have a folder named after the dataset we are using for example "OMML-100K" which consists of "images" folder and "original_mml.lst" file.

```
cd skema/img2mml; mkdir training_data
```

To prepare the dataset for training, we will first preprocess it. The first step will be the image preprocessing where all the empty and corrupted images will be removed. The second step will be preprocessing the target MathML equations where again all the equations corresponding to rejected images will be removed from the dataset and all equations which have less than 2 tokens will be rejected as well since a MathML equations can't be defined in just 2 tokens. An example can be seen as `<math> </math>` which doesn't represent anything.

#### To preprocess images:
```
python preprocessing/preprocess_images.py --config configs/xfmer_mml_config.json
```

#### To preprocess mathml:
```
python preprocessing/preprocess_mml.py --config configs/xfmer_mml_config.json
```

#### To train the model:
```
python training.py --config configs/xfmer_mml_config.json
```

#### To calculate Torchtext BLEU score
```
python utils/bleu_score.py
```

For multiperl bleu score:
```
perl utils/multi-bleu.perl logs/final_targets.txt < logs/final_preds.txt"
```

#### To calculate edit distance
```
python utils/edit_distance.py
```

## Inference

The model itself is not checked into the repository, but you can get it from
here:
https://kraken.sista.arizona.edu/skema/img2mml/models/cnn_xfmer_OMML-90K_best.pt.

Place the model file in the `trained_models` directory.

The curl command below should do the trick.

```
curl -L https://kraken.sista.arizona.edu/skema/img2mml/trained_models/cnn_xfmer_OMML-90K_best.pt > trained_models/cnn_xfmer_OMML-90K_best.pt
```

If model is trained on different server, then `vocab.txt` file need to transferred here.
```
curl -L https://kraken.sista.arizona.edu/skema/img2mml/vocab.txt > vocab.txt
```

Then, run the invocation below to launch the Dockerized service:

```
docker-compose up --build
```

To test the service without Docker (e.g., for development purposes), ensure
that you have installed the required packages (run `pip install -e .[img2mml]`
in the root of the repository. In case of Zsh, run `pip install -e ."[img2mml]"`).

Then, run the following command to launch the img2mml server program:

```
uvicorn img2mml:app --reload
```

An example test program is provided as well, which you can invoke with:
Make sure that you are in generate_mathml folder.

```
python img2mml_demo.py
```
