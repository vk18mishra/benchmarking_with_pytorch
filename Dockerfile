FROM pytorch/pytorch:latest
COPY . /usr/benchmarkpytorch/
EXPOSE 5000
WORKDIR /usr/benchmarkpytorch/
RUN python3 --version
RUN python3 --version
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install git+https://github.com/powerapi-ng/pyRAPL.git#egg=pyRAPL
RUN pip3 install -r requirements.txt