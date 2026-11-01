FROM rocm/pytorch:latest

WORKDIR /app

RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers" \
    transformers \
    accelerate \
    pillow \
    torchao

# API server: FastAPI, uvicorn, Pillow, PyYAML, python-multipart, MCP SDK.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The code is copied in at build time: editing server/ needs `docker compose up -d --build`.
COPY server/ /app/server/

RUN chmod -R a+rX /app

EXPOSE 8000

CMD ["python3", "-m", "server"]
