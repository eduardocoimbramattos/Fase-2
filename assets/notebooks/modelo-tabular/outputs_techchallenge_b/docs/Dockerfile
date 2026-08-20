FROM python:3.11-slim

WORKDIR /app
COPY techchallengeB.py /app/techchallengeB.py
COPY data.csv /app/data.csv

RUN pip install --no-cache-dir pandas numpy matplotlib scikit-learn shap

CMD ["python", "techchallengeB.py", "--data", "/app/data.csv", "--output", "/app/outputs"]
