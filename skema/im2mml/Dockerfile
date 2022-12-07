FROM python:3.8

WORKDIR /code/generate_mathml

COPY ./generate_mathml/requirements.txt /code/generate_mathml

RUN pip install -r requirements.txt

COPY ./generate_mathml /code/generate_mathml
