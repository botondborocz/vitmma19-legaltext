# 1. Base Image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 2. Munkakönyvtár
WORKDIR /app

# 3. Python csomagok
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. MÓDOSÍTÁS ITT: Az src mappát src mappába másoljuk!
# Így megmarad a szerkezet: /app/src/train.py
COPY ./src ./src

# 5. Jogosultság a run.sh-nak (most már az src-ben van)
RUN chmod +x src/run.sh

RUN mkdir -p log output

# 6. Az indítási útvonal
CMD ["bash", "src/run.sh"]