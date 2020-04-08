FROM python:3.8.2-slim

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -qq update

ADD . .
RUN pip install -qr requirements.txt

ENTRYPOINT ["python", "train.py"]
CMD ["--help"]
