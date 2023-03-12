# Sample data

Create dataset for training purpose from the raw dataset that has been generated from the arxiv source files. For further details about data generation, check out `data_generation/README.md`.

### Distribution
We need to provide the distribution which we want to follow while sampling our dataset for training purpose. To do this, update the `sampling_dataset/sampling_config.json`.

The dataset distribution is largely divided into a few bin based on the length of the MathML equations. These bins are:
[0,50], [50,100], [100,150], [150,200], [200,250], [250,300], [300,350], [350,1000]. In every bin represents range of the number of tokens a MathML equation can have.

To sample dataset from the year(s) i.e. considering all months in that year(s), set `sample_from_years` to true in the sampling_config file. To sample from specific month folders, set `sample_from_years` to false and `sample_from_months` to true.

The `src_path` will be the destination path of the `data_generation/data_generation_config.json`.
