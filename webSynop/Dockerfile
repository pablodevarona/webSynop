FROM python:3.11-alpine

WORKDIR /app

COPY requirements.txt pyodbc.sh ./

RUN apk update && apk add gnupg curl g++ unixodbc-dev

RUN /bin/sh pyodbc.sh

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "/app/inicio.py", "--server.port=8501", "--server.address=0.0.0.0"]