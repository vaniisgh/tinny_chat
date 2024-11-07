## Tinny QA bot

<p align="center">
  <img src="./docs/logo.jpg" alt="Logo" width="200"/>
</p>

A completely local solution to querying your doccuments, it has a single endpoint
    - /query_document
that helps you query a doccument using a llm ( by default tinyllama), you can modify
the llm and prompt from the main.py file.

use the /docs endpoint of fastapi to see the required input and output parameters.

Use any PDF file, or a JSON in the example format in [sample_data.json](./sample_data.json) as the input document, and send the questions in the [sample_questions.json](./sample_questions.json) format.


Note: every query runs a separate `invoke` for the model, more queries 
slower response and pulling the model also takes some time.


### Run the dockerised application

```
docker compose up --build -d
```

for more instructions refer to [screenshots](./additional_screenshots.md)

## Running without docker

You need the following services to run the QA bot:
- Chroma DB
- Ollama Docker container
- This service

You  can use the steps below to start the server

```shell
cd path/to/this/repo
pip install -r requirements.txt
```

reanme [./confing_no_docker.ini] to ./config.ini to run wihtout docker compose

run chromadb, if you change the port please chage in the config.ini[./config.ini]
file as well.

```shell
chroma run --host localhost --port 8000 --path ./my_chroma_data
```

run the current app
```
fastapi dev main.py --port 8080 --reload
```

start your docker desktop and then run the follwoing to
get your ollama ready !
```shell
docker pull ollama/ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker ps
```
copy the name of the docker container

```shell
docker exec -it [container-name] bash
ollama pull tinyllama && exit
```
