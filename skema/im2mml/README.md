# im2mml

This directory contains the code for the im2mml service, which processes images
of equations and returns presentation MathML corresponding to those equations.

The model was developed by Gaurav Sharma and Clay Morrison, and this wrapper
service was developed by Deepsana Shahi and Adarsh Pyarelal.

The model itself is not checked into the repository, but you can get it from
here: https://kraken.sista.arizona.edu/skema/im2mml/models/opennmt_ourmml_100K_lte100_best.pt

Place the model file in the `generate_mathml` directory, then run the
invocation below to launch the Dockerized service:

```
docker-compose up --build
```
