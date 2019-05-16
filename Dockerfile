FROM ubuntu:16.04
FROM python:3.6

RUN apt-get update
ADD . /root/estimate_price/

WORKDIR /root/estimate_price/

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "api:api"]
