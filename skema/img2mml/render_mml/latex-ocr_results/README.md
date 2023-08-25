# Pix2TeX Model Results

## Predictions

The predictions were made using this [pix2tex model](https://github.com/lukas-blecher/LaTeX-OCR). My script called
[predict.py](./predict.py) was used to generate the predictions.

- The results are stored in [im2latex_100k_test_results.json](./img2latex_100k_test_results.json).
- There were 30 errors for which the model failed to make predictions; these are stored in [im2latex_100k_test_errors.json](./img2latex_100k_test_errors.json).

## Data

The predict script expects uses data from the [im2latex100k](https://im2markup.yuntiandeng.com/data/) given in the original repository. The model makes predictions for everything in `im2latex_test.lst` and compares it to the ground truth.

This dataset is different from what is currently linked in the repository because it appears that the model in pix2tex was actually trained on the normalized copy of the dataset which I have linked above.

## Running Predictions

- Clone the [LatexOCR repo](https://github.com/lukas-blecher/LaTeX-OCR) and follow the instructions to install the dependencies and the model.
- Collect the data as outline in the data section below and put it in the cloned repo.

  - The data should be in the same directory as the `predict.py` script.
  - The file structure should be:

  ```
  cloned_repo
  ├── predict.py
  ├── im2latex100k
  │   ├── formula_images
  │   ├── im2latex_test.lst
  │   ├── im2latex_formulas.norm.lst
  ```

- Copy and run `python predict.py` in the repo to generate the predictions.
