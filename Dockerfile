FROM rocm/pytorch:latest

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    "git+https://github.com/huggingface/diffusers" \
    transformers \
    accelerate \
    pillow \
    torchao \
    pyyaml

COPY server/ /app/

RUN chmod -R a+rX /app

EXPOSE 8000

CMD ["python3", "main.py"]