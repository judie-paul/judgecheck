FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JUDGECHECK_REPORTS=/data/reports
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir '.[app,llm,hf]' \
    && useradd --create-home judge \
    && mkdir -p /data/reports /data/cache \
    && chown -R judge /data
USER judge
ENV JUDGECHECK_CACHE_DIR=/data/cache
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --retries=6 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["judgecheck", "app", "--address", "0.0.0.0", "--port", "8501"]
