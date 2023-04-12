FROM jupyter/scipy-notebook

USER root

RUN apt-get update && apt-get install -y \
	cmake \
	libgtk2.0-dev \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev



COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN chown -R jovyan:users /home/jovyan
RUN chown -R $NB_UID /home/$NB_USER
USER $NB_USER

