# Data Generation Scripts


### Updates paths in the config files

-src (source_path) = path to the directory where the data has been stored.  

-dst (destination path) = path to the destination directory.  

-dir (List of Directories) = for example: 1401 1402 1403  

-yr (year) = for example: 2014

### Requirements

```
python3 -m pip install -r data_gen_requirements.txt
```

After installing requirements, a few more packages will be required to run mathjax-server that can be installed using following commnads.

Check if node is properly installed. If in case it is not installed, try using
```
conda install -c conda-forge nodejs
```
```
npm install mathjax-node
```
```
npm install express
```

Also, for TeX to PDF to PNG conversion, `pdflatex` and `ImageMagick` has to be installed on the system.

### Scripts

This repository is for generating the dataset from raw arXiV latex source. It has scripts that will do the following jobs:

1) parse the mathematical equations from the raw arXiV latex source paper(s). All symbols, greek letters are provided in the `latex_symbols.xlsx` file.
```
python parsing_latex_equations.py
```

2) After getting the mathematical equations, we will create TeX files for each equation which will be used to render the PDF and PNG image of the equation.
```
python tex_builder.py
```
```
python tex2png.py
```

3) Render MathML from the latex equations. NOTE: We first need to run MathJax or TeMML server in a separate terminal and then will proceed with the MathML rendering script.
```
node mathjax_server.js
```
if using Temml service, run temml service:
```
node temml_service.js
```

```
python latex2mathml.py
```