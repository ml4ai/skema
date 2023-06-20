
Documentation is built using [`mkdocs`, a static site generator written in Python](https://www.mkdocs.org/).

# Building the docs
Assuming you've [configured your development enviroment](../env/):

```bash
conda activate skema
mkdocs serve
```

Open your browser to [http://127.0.0.1:8000/skema](http://127.0.0.1:8000/skema).

!!! note
    `mkdocs serve` supports **live reloading**.  Any changes to the source will be reflected immediately. 



# Adding navigation links

Site navigation is defined in [`mkdocs.yml`](https://github.com/ml4ai/skema/blob/main/mkdocs.yml) and references markdown and html pages located under `docs`.