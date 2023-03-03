# Data Generation Scripts

Packages to install: xlrd==1.2.0, openpyxl

### Updates paths in the config files

-src (source_path) = path to the directory where the data has been stored.  

-dst (destination path) = path to the destination directory.  

-dir (List of Directories) = for example: 1401 1402 1403  

-yr (year) = for example: 2014


### Scripts

This repository is for generating the dataset from raw arXiV latex source. It has scripts that will do the following jobs:

1) parse the mathematical equations from the raw arXiV latex source paper(s). All symbols, greek letters are provided in the _Latex_symbols.xlsx_ file.
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
node mathJax_server.js
```
if using Temml service, run temml service:
```
node temml_service.js
```

```
python latex_mathml.py
```

4) Simplify the MathML equations by removing unnecessary details or tokens. Those simplified MathML equations will then be used to render XML ElementTrees using pretty print.
```
python mathml_simplification.py
```
```
python etreeParser.py
```
