import os
import json
import configparser
import logging

from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager


import chromadb
from chromadb.config import Settings

from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter

from langchain_ollama import ChatOllama

from models import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")

CHROMA_HOST = config["CHROMA"]["HOST"]
CHROMA_PORT = int(config["CHROMA"]["PORT"])
CHROMA_COLLECTION_NAME = config["CHROMA"]["COLLECTION_NAME"]

OLLAMA_HOST = config["OLLAMA"]["HOST"]

llm_components = {}


def setup_chroma_client():
    """
    Initialize and configure the Chroma client with settings from config

    Returns:
        chromadb.HttpClient: Configured Chroma client instance
    """
    logger.info("Setting up Chroma client")
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings()
    )


def setup_llm_model():
    """
    Initialize the LLM model with Ollama configuration

    Returns:
        ChatOllama: Configured Ollama chat model instance
    """
    logger.info("Setting up LLM model")
    return ChatOllama(
        model="tinyllama",
        base_url=f"http://{OLLAMA_HOST}:11434",
        verbose=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of the FastAPI application

    Args:
        app (FastAPI): The FastAPI application instance
    """
    logger.info("Starting application lifecycle")
    chroma_client = setup_chroma_client()
    model = setup_llm_model()

    llm_components['model'] = model
    llm_components['chroma_client'] = chroma_client

    logger.info(f"Created collection {CHROMA_COLLECTION_NAME}")

    yield

    logger.info("Cleaning up resources")
    chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
    logger.info(f"Deleted collection {CHROMA_COLLECTION_NAME}")


async def process_pdf_file(file_content, filename):
    """
    Process a PDF file by splitting it into manageable chunks

    Args:
        file_content: Binary content of the PDF file
        filename (str): Name of the PDF file

    Returns:
        list: List of chunked documents from the PDF
    """
    logger.info(f"Processing PDF file: {filename}")
    file_path = f"/tmp/{filename}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = PyPDFLoader(file_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(document)

    os.remove(file_path)
    logger.info(f"Successfully processed PDF file: {filename}")
    return chunked_documents


async def process_json_file(file_content):
    """
    Process a JSON file and split it into documents

    Args:
        file_content: Binary content of the JSON file

    Returns:
        list: List of documents created from JSON content, None if invalid format
    """
    logger.info("Processing JSON file")
    document = json.loads(file_content.decode("utf-8"))
    if type(document) is not dict:
        logger.error("JSON file does not contain a dictionary")
        return None
    json_splitter = RecursiveJsonSplitter()
    logger.info("Successfully processed JSON file")
    return json_splitter.create_documents(texts=[document])


def embed_documents(chunked_documents, chroma_client):
    """
    Embed documents using sentence transformers and store in Chroma

    Args:
        chunked_documents (list): List of documents to embed
        chroma_client: Chroma client instance

    Returns:
        int: Number of documents embedded
    """
    logger.info("Embedding documents")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=chunked_documents,
        embedding=embedding_function,
        collection_name=CHROMA_COLLECTION_NAME,
        client=chroma_client,
    )
    logger.info(f"Embedded {len(chunked_documents)} documents")
    return len(chunked_documents)


def process_questions(question_list, documents, model):
    """
    Process a list of questions using the provided documents and model

    Args:
        question_list (list): List of questions to process
        documents (list): List of relevant documents for context
        model: LLM model instance

    Returns:
        dict: Dictionary mapping questions to their answers
    """
    logger.info(f"Processing {len(question_list)} questions")
    prompt = """
    You are a helpful assistant, the user has asked you {question}.
    And you have to answer the question based on the following details:
    {retreived_documents}, if you have no context please return "I don't know".
    Also please try to anser the question only and not provide any additional information
    about youself please, this is an important answer.
    """

    results = dict()
    for question in question_list:
        logger.debug(f"Processing question: {question}")
        this_prompt = prompt.format(
            question=question,
            retreived_documents="".join([f"{i+1}. {result}\n" for i,
                                         result in enumerate(documents)])
        )
        response_message = model.invoke(this_prompt)
        results[question] = response_message.content

    logger.info("Completed processing all questions")
    return results


app = FastAPI(lifespan=lifespan)


@app.post("/documents/")
async def add_document(file: UploadFile = File(...), questions: UploadFile = File(...)) -> Response:
    """
    FastAPI endpoint to process uploaded documents and answer questions

    Args:
        file (UploadFile): Uploaded document file (PDF or JSON)
        questions (UploadFile): Uploaded file containing questions

    Returns:
        Response: Object containing answers to questions or error message
    """
    logger.info(f"Received document upload request: {file.filename}")
    file_content = await file.read()
    file_type = file.content_type

    question_content = await questions.read()
    questions_json = json.loads(question_content.decode("utf-8"))

    if file_type == "application/pdf":
        chunked_documents = await process_pdf_file(file_content, file.filename)
    elif file_type == "application/json":
        chunked_documents = await process_json_file(file_content)
        if chunked_documents is None:
            logger.error("Invalid JSON document format")
            return Response(root={"error": "JSON file must contain a dictionary"})
    else:
        logger.error(f"Unsupported file type: {file_type}")
        return Response(root={"error": "Unsupported file type"})

    chroma_client = llm_components['chroma_client']
    chunks_count = embed_documents(chunked_documents, chroma_client)
    logger.info(f"Added {chunks_count} chunks to chroma db")

    collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
    chroma_recollection = collection.query(
        query_texts=questions_json,
        n_results=len(questions_json)
    )

    documents = chroma_recollection['documents']
    model = llm_components['model']
    results = process_questions(questions_json, documents, model)

    logger.info("Successfully processed document and questions")
    return Response(root=results)
