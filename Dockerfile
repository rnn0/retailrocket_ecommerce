FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# A biblioteca `implicit` nao tem wheel pronto para esta imagem: precisa de compilador para instalar.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY docker ./docker

# O treino NÃO acontece no build (dados são grandes e ficam fora da imagem):
# o entrypoint treina na primeira subida se models/ estiver vazio.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
ENTRYPOINT ["./docker/entrypoint.sh"]
