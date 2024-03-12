FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04




WORKDIR /root

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN pip install --no-cache --upgrade pip setuptools
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt




COPY . .


ENTRYPOINT [ "python3.10", "main.py" ] 

