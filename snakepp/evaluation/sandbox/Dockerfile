# TODO (Enea): pin the DS-1000 library versions in versions.txt
FROM python:3.10-slim
RUN useradd -m runner
COPY versions.txt /tmp/versions.txt
RUN pip install --no-cache-dir -r /tmp/versions.txt
USER runner
WORKDIR /home/runner
