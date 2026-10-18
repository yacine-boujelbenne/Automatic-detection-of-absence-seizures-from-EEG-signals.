# Interface Streamlit dans un conteneur. Les données ne sont pas copiées dans l'image :
#   docker build -t absences-eeg .
#   docker run --rm -p 8501:8501 \
#     -v "$PWD/data:/app/data:ro" -v "$PWD/artifacts:/app/artifacts:ro" -v "$PWD/app/model:/app/app/model" \
#     absences-eeg
# puis ouvrir http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY tools ./tools
COPY scripts ./scripts
COPY .streamlit ./.streamlit
COPY ["Annotations MAJ.xlsx", "./"]

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
