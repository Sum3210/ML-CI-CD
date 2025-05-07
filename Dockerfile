# Używamy oficjalnego obrazu Pythona 3.9 slim
FROM python:3.9-slim

# Ustawiamy katalog roboczy w kontenerze
WORKDIR /app

# Kopiujemy requirements.txt i instalujemy zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy plik aplikacji
COPY app.py .

# Uruchamiamy serwer Flask
CMD ["python", "app.py"]