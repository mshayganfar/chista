FROM python:3.10

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w 2", "-b", "0.0.0.0:5000", "app:app"]