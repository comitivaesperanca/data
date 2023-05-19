FROM apache/airflow:2.5.2
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

  
# Install Google Chrome
RUN apt-get -y update && apt-get install -y wget curl\
&& wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
&& sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
&& apt-get -y update \
&& apt-get install -y google-chrome-stable xvfb

# Install Chrome Driver
RUN apt-get install -yqq unzip \
&& wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip \
&& unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/ \
&& apt-get -y --purge autoremove

RUN apt-get install -y default-libmysqlclient-dev && apt-get install -y mysql-client 


RUN chmod -R 777 /opt/airflow/logs
ARG AIRFLOW_UID=50000
COPY requirements.txt requirements.txt
USER airflow
RUN pip install --upgrade pip
RUN pip install -r requirements.txt