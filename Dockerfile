FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt
RUN pip install gunicorn

COPY ./api /code/api

EXPOSE 8000

# CMD ["gunicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8080"]

ENV FLASK_APP=api/index.py

# Command to run the Flask app with gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "api.index:app"]



