FROM python:3.8

WORKDIR /code/generate_mathml

COPY ./generate_mathml/requirements.txt /code/generate_mathml

RUN pip install -r requirements.txt

COPY ./generate_mathml /code/generate_mathml
#COPY ./translate.py /code/generate_mathml

EXPOSE 8000

CMD ["uvicorn", "translate:app", "--host=0.0.0.0", "--reload"]


#CMD ["uvicorn", "generate_mathml.translate:app", "--host=0.0.0.0", "--reload"]
#CMD ["uvicorn", "generate_mathml.translate:app", "--host", "0.0.0.0", "--port", "80"]
