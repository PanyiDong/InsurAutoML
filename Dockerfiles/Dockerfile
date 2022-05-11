# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim-buster
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update pip
RUN python -m pip install --upgrade pip

# Install git
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install git

# Install pip requirements
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Clone AutoML from git to workspace
RUN git clone https://github.com/PanyiDong/My_AutoML.git /app/My_AutoML
RUN cd /app/My_AutoML
# Install dependencies
RUN pip install -e .[normal]

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py --data_folder example/example_data --train_data heart --response HeartDisease"]
