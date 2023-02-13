Scripts:
===========================

To parse the mathematical equations from the paper(s). All the symbols, greek letters are provided in the _Latex_symbols.xlsx_ file which will be called in _parsing_latex_equations.py_
```
python parsing_latex_equations.py -src </path/to/arxiv/papers/>  -dst </path/to/destination/of/parsed_equations/>  -yr </path/to/year/folder/> -dir </path/to/specific/month/directory/>
```
After getting the mathematical equations, we need to make TeX files for each equation which then will be used to render the PDF and PNG image of that equation.
```
python tex_builder.py -src </path/to/arxiv/papers/>  -dst </path/to/destination/of/parsed_equations/>  -yr </path/to/year/folder/> -dir </path/to/specific/month/directory/>
```
```
python tex2png.py -src </path/to/arxiv/papers/>  -dst </path/to/destination/of/parsed_equations/>  -yr </path/to/year/folder/> -dir </path/to/specific/month/directory/>
```
Now it's time to render MathML from the latex equations we've parsed. 
NOTE: We first need to run MathJax server in a separate terminal and then will proceed with the MathML rendering script.
```
node MathJax_server.js
```
```
python latex_mathml.py -src </path/to/arxiv/papers/>  -dst </path/to/destination/of/parsed_equations/>  -yr </path/to/year/folder/> -dir </path/to/specific/month/directory/>
```

Once we got MathML codes for all the equations, we will simplify them by removing unnecessary/not-important details. Those simplified MathML equations will then be used to render XML ElementTrees using pretty print.
```
python mathml_simplification.py -src </path/to/arxiv/papers/>  -dst </path/to/destination/of/parsed_equations/>  -yr </path/to/year/folder/> -dir </path/to/specific/month/directory/>
```
```
python etreeParser.py -src </path/to/arxiv/papers/>  -dst </path/to/destination/of/parsed_equations/>  -yr </path/to/year/folder/> -dir </path/to/specific/month/directory/>
```

# Paths need to provide for venti(local server)

-src (source_path) = '/projects/temporary/automates/arxiv/src'

-dst (destination path) = '/projects/temporary/automates/er/gaurav' 

-dir (List of Directories) = for example: 1401 1402 1403  

-yr (year) = for example: 2014


# Paths need to provide for HPC

-src (source_path) = '/xdisk/claytonm/projects/automates/arxiv/src'

-dst (destination path) = '/xdisk/claytonm/projects/automates/er/gaurav' 

-dir (List of Directories) = for example: 1401 1402 1403

-yr (year) = for example: 2014
