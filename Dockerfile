FROM pytorch/pytorch:latest
COPY . /usr/benchmarkpytorch/
EXPOSE 5000
WORKDIR /usr/benchmarkpytorch/
RUN python3 --version
RUN python3 --version
# RUN pip install --upgrade pip
# RUN pip install --upgrade setuptools
# RUN pip install setuptools gitpython
RUN pip3 install pyRAPL
RUN pip3 install -r requirements.txt
