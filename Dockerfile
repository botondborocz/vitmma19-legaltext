# 1. Base Image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 2. Workdir
WORKDIR /app

# 3. Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Code
COPY ./src ./src

# 5. Permissions
RUN chmod +x src/run.sh
RUN mkdir -p data log output

# 6. Run
CMD ["bash", "src/run.sh"]