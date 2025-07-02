FROM python:3.11-slim

# 1. install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. copy source
COPY vmc_model/ vmc_model/
COPY config.yaml .
COPY README.md ./

ENTRYPOINT ["python", "-m", "vmc_model.cli"]
