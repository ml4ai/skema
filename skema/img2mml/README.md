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
We will sample the dataset from the raw dataset that we have generated in above step.

## Training

Before starting the training, please ensure that the dataset is at `skema/skema/img2mml/training_data` folder. The `training_data` folder will have a folder named after the dataset we are using for example "OMML-100K" which consists of "images" folder and "original_mml.lst" file.

```
cd skema/img2mml; mkdir training_data
```

To prepare the dataset for training, we will first preprocess it. The first step will be the image preprocessing where all the empty and corrupted images will be removed. The second step will be preprocessing the target MathML equations where again all the equations corresponding to rejected images will be removed from the dataset and all equations which have less than 2 tokens will be rejected as well since a MathML equations can't be defined in just 2 tokens. An example can be seen as `<math> </math>` which doesn't represent anything.

#### To preprocess images:
```
python preprocessing/preprocess_images.py --config configs/xfmer_config.json
```

#### To preprocess mathml:
```
python preprocessing/preprocess_mml.py
```

#### To train the model:
```
python training.py configs/xfmer_config.json
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
