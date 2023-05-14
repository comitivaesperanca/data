FROM python:3.9

# 
WORKDIR /app

# 
COPY ./requirements.txt /app/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 
COPY ./app /app/

# 
CMD ["uvicorn", "main:app","--reload", "--reload-include", ".*, .py[cod]", "--host", "0.0.0.0", "--port", "7000"]