import os
import json
import configparser

from fastapi import FastAPI, UploadFile, File, Query
from contextlib import asynccontextmanager

from typing import List, Annotated

import chromadb
from chromadb.config import Settings

from langchain_chroma import Chroma


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter

from langchain_ollama import ChatOllama


from models import Questions, Response

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")

CHROMA_HOST = config["CHROMA"]["HOST"]
CHROMA_PORT = int(config["CHROMA"]["PORT"])
CHROMA_COLLECTION_NAME = config["CHROMA"]["COLLECTION_NAME"]

OLLAMA_HOST = config["OLLAMA"]["HOST"]

llm_components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    chroma_client = chromadb.Client()
    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings()
    )

    model = ChatOllama(
        model="tinyllama",
        base_url=f"http://{OLLAMA_HOST}:11434",
        verbose=True,
    )

    llm_components['model'] = model
    llm_components['chroma_client'] = chroma_client

    print(f"Created collection {CHROMA_COLLECTION_NAME}")

    yield

    print("Cleaning up")
    chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
    print(f"Deleted collection {CHROMA_COLLECTION_NAME}")

app = FastAPI(lifespan=lifespan)


@app.post("/documents/")
async def add_document(file: UploadFile = File(...), questions: UploadFile = File(...)) -> Response:

    # Process the file (PDF or JSON)
    file_content = await file.read()
    file_type = file.content_type

    # Currently cant get FASTAPI to accept them as a list in the multipart form
    question_content = await questions.read()
    questions_json = json.loads(question_content.decode("utf-8"))

    if file_type == "application/pdf":
        # Process PDF file directly from the uploaded content
        file_path = f"/tmp/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(file_content)

        # Process PDF file
        loader = PyPDFLoader(file_path)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents(document)

        # Clean up the temporary file
        os.remove(file_path)
    elif file_type == "application/json":
        # Process JSON file
        document = json.loads(file_content.decode("utf-8"))
        if type(document) is not dict:
            return Response(root={"error": "JSON file must contain a dictionary"})
        json_splitter = RecursiveJsonSplitter()
        chunked_documents = json_splitter.create_documents(texts=[document])

    else:
        return Response(root={"error": "Unsupported file type"})

    # Embed and index the documents
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_client = llm_components['chroma_client']
    Chroma.from_documents(
        documents=chunked_documents,
        embedding=embedding_function,
        collection_name=CHROMA_COLLECTION_NAME,
        client=chroma_client,
    )

    print(f"Added {len(chunked_documents)} chunks to chroma db")

    # Process the list of questions
    question_list = questions_json

    collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
    chroma_recollection = collection.query(
        query_texts=question_list,
        n_results=len(question_list)
    )

    # chromadb always returns a list, no need to handle error here
    documents = chroma_recollection['documents']

    model = llm_components['model']
    prompt = """
    You are a helpful assistant, the user has asked you {question}.
    And you have to answer the question based on the following details:
    {retreived_documents}, if you have no context please return "I don't know".
    Also please try to anser the question only and not provide any additional information
    about youself please, this is an important answer.
    """

    results = dict()
    for question in question_list:
        this_prompt = prompt.format(
            question=question,
            retreived_documents="".join([f"{i+1}. {result}\n" for i,
                                         result in enumerate(documents)])
        )

        response_message = model.invoke(
            this_prompt
        )
        results[question] = response_message.content

    return Response(root=results)
