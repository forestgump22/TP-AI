FROM python:3.11.5

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY conexion.py . 
COPY Usuario.py .   
COPY templates/ ./templates/  
COPY static/ ./static/
COPY .env . 

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
