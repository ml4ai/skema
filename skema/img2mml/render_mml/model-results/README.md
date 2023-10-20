## Model Results

This section contains the results of the model. The results are presented in the form of a JSON file and graphs available in the `results` folder.

### Using

1. Create virtual environment and install requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1. Run the following command to generate the result statistics from the model's outputs.

```bash
./eval
```

The results will be saved in the `results` folder.

3. (Optional) To visualize the results, run the following command

```bash
./vis
```

The images will be saved in the `results/images` folder.
