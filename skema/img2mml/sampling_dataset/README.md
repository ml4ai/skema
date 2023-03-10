# Sample data

Create dataset for training purpose from the raw dataset that has been generated from the arxiv source files. For further details about data generation, check out _data_generation/README.md_.

### Distribution
We need to provide the distribution which we want to follow while sampling our dataset for training purpose. To do this, update the sample_data/_sampling_config.json_.

The dataset distribution is largely divided into a few bin based on the length of the MathML equations. These bins are:
[0,50], [50,100], [100,150], [150,200], [200,250], [250,300], [300,350], [350,1000]. In every bin represents range of the number of tokens a MathML equation can have.

To sample dataset from the year(s) i.e. considering all months in that year(s), set _sample_from_years_ to true in the sampling_config file. To sample from specific month folders, set _sample_from_years_ to false and _sample_from_months_ to true.
