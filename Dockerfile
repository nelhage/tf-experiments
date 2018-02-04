FROM us.gcr.io/livegrep/tf-base:latest

RUN virtualenv --python=python3 /venv

RUN mkdir /src
ADD requirements.txt /src/requirements.txt

WORKDIR /src

RUN /venv/bin/pip install -r requirements.txt

ADD . /src
