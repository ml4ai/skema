FROM python python:3.8

WORKDIR /skema/skema/app_generate_mathml

COPY ./requirements.txt

RUN pip install -r requirements.txt

COPY ./generate_mathml /skema/skema/app_generate_mathml/generate_mathml

CMD ["uvicorn", "generate_mathml.translate:app", "--host", "0.0.0.0", "--port", "80"]
