FROM amd64/python:3.9.12

COPY kubernetes/requirements.txt requirements.txt

RUN pip install -r requirements.txt && mkdir data

COPY stage1.py stage1.py

ADD data_preprocessed /data_preprocessed

CMD ["python", "stage1.py"]