FROM python:3.7

WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY app.py ./app.py
COPY inference.py ./inference.py
COPY __init__.py ./__init__.py
ADD ./output ./output
ADD ./src ./src
ADD ./input ./input
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["./app.py"]
