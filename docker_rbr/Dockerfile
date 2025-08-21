FROM --platform=linux/amd64 tensorflow/tensorflow:2.18.0-gpu AS example-algorithm-amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    unzip \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user resources/target.png /opt/app/resources/
COPY --chown=user:user resources/python_2D_versatile_he.zip /opt/app/resourcescd/
COPY --chown=user:user model /opt/app/model

RUN mkdir -p /home/user/.keras/models/StarDist2D/2D_versatile_he && \
    unzip /opt/app/resources/python_2D_versatile_he.zip -d /home/user/.keras/models/StarDist2D/2D_versatile_he/2D_versatile_he && \
    unzip /opt/app/resources/python_2D_versatile_he.zip -d /home/user/.keras/models/StarDist2D/2D_versatile_he/2D_versatile_he_extracted && \
    mv /opt/app/resources/python_2D_versatile_he.zip /home/user/.keras/models/StarDist2D/2D_versatile_he/2D_versatile_he.zip

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user rbr.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
