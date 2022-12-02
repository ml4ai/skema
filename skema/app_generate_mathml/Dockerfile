FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code

RUN pip install -r requirements.txt

COPY ./generate_mathml /code/generate_mathml

EXPOSE 8000

CMD ["uvicorn", "generate_mathml.translate:app", "--host=0.0.0.0", "--reload"]

#CMD ["uvicorn", "generate_mathml.translate:app", "--host", "0.0.0.0", "--port", "80"]
