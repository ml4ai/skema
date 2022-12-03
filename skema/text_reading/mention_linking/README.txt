Chime SIR
Gromet file: https://drive.google.com/file/d/14EYhTLsCbUZhDMrczlOdfDADcn2jzgLG/view?usp=share_link

https://drive.google.com/file/d/1Xe3-97e3Sk3uVVlPyYkGKLPDXtH6P8ny/view?usp=share_link

Improve the grammars

Do the same for CHIME_SVIIvR

If there is time test full penn, bucky model

Pointer to locations with code for each model
https://docs.google.com/document/d/1vUsC_YRw3ddewOFYGzcsi4QH3C0x4cLnBL4EU5q5DO8/edit#


Grounding Pipeline:
0. Identify the papers for each model/source code
1. Generate the Grommet representation of the source code (py_src_to_grometFN_JSON.ipynb)
2. Extract the comments of the source code (Skema TextReading)
3. Write script to link comments to OFP of grommet
4. Write script to do text similarity to ParameterAndValue extractions




Progress:
We have been doing work on two fronts:
Collecting a corpus a code comments based on a targeted search for epidemiology repositories in GitHub based on queries derived from MITRE's codev wiki
Building the comment linking script to
