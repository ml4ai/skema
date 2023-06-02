# img2mml

This directory contains the code for the img2mml service, which processes images
of equations and returns presentation MathML corresponding to those equations.

The model was developed by Gaurav Sharma and Clay Morrison, and this wrapper
service was developed by Deepsana Shahi and Adarsh Pyarelal.

## Building virtual environment and installling requirements

### Using pip

We are using `pyenv` to create pip virtual environment. Feel free to skip
this step if have python installed.

```
pip install pyenv

pyenv virtualenv 3.8.16 image2math_venv

python3 -m pip install -r requirements.txt
````

### Using conda virtual environment
```
conda create -n image2mathml_venv python=3.8 -y ; conda activate image2mathml_venv

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y

conda install --file requirements.txt
```

To deactivate (if used pyenv to create virtual environment)
```
pyenv deactivate
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

### Preparing config file:
The config file can be found under `configs/`. Based on the dataset, select the respective config file. Parameters that need to set before training are as follows:

`model_type`: resnet_xfmer, cnn_xfmer, opennmt (a local replica of OpenNMT).

`device`:cpu or cuda. If using Cuda, please modify the GPU related parameters in the config file.

Please ensure that `data_path` and `dataset_type` are properly set under `params for preprocessing`.

If only want to test the pre-trained model, change the `load_trained_model_for_testing` to true.

#### To preprocess images:
```
python preprocessing/preprocess_images.py --config configs/xfmer_mml_config.json
```

#### To preprocess mathml:
```
python preprocessing/preprocess_mml.py --config configs/xfmer_mml_config.json
```

#### To train and test the model:

To train, set `testing=false` while to test set it to `true`.
```
python training.py --config configs/xfmer_mml_config.json
```

#### To calculate Torchtext BLEU score
```
python utils/bleu_score.py
```

For multiperl bleu score:
```
perl utils/multi-bleu.perl logs/final_targets.txt < logs/final_preds.txt
```

#### To calculate edit distance
```
python utils/edit_distance.py
```

## Inference

If not training and directly want to use the model for translating the image, in that case please get the trained model, and the vocab file from the server.
Place the trained model file in the `trained_models` directory and the `vocab.txt` file under `img2mml`

The curl commands below should do the trick.

```
curl -L https://kraken.sista.arizona.edu/skema/img2mml/trained_models/cnn_xfmer_omml-100K_best.pt > trained_models/cnn_xfmer_omml-100K_best.pt

curl -L https://kraken.sista.arizona.edu/skema/img2mml/vocab.txt > vocab.txt
```

Then to run the invocation below to launch the Dockerized service:
```
docker-compose up --build
```

Then, run the following command to launch the img2mml server program:

```
uvicorn img2mml:app --reload
```

An example test program is provided as well, which you can invoke with:
Make sure that you are in generate_mathml folder.

```
python img2mml_demo.py
```
