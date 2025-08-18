FROM --platform=linux/amd64 tensorflow/tensorflow:latest-gpu AS example-algorithm-amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

#RUN groupadd -r user && useradd -m --no-log-init -r -g user user
#USER user

WORKDIR /opt/app

COPY requirements.txt /opt/app/
COPY resources /opt/app/resources
COPY resources/target.png /opt/app/resources/

COPY model /opt/app/model

RUN apt-get update && apt-get install -y libgl1

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
