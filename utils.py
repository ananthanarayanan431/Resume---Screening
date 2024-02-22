import openai

import os
from constant import openai,PINECONE_API_KEY

os.environ['OPENAI_API_KEY'] = openai
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub
from langchain_community.vectorstores import Pinecone as PineconeLangchain

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_text_pdf(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def create_docs(user_pdf_list, unique_id):
    docs = []
    val = 0
    for filename in user_pdf_list:
        chunks = get_text_pdf(filename)

        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name, "id": val, "type=": filename.type, "size": filename.size,
                      "unique_id": unique_id},
        ))
        val += 1

    return docs


def create_embedding_load_data():
    embeddings = OpenAIEmbeddings()
    # embeddings = SentenceTransformerEmbeddings(
    #     model_name="all-MiniLM-L6-v2"
    # )
    return embeddings


def push_to_pinecone(pinecone_index_name, embeddings, docs):

    index = pinecone_index_name
    print("Hello World")
    PineconeLangchain.from_documents(docs, embeddings, index_name=index)


def pull_to_pinecone(embeddings):
    # pinecone.init(
    #     api_key=pinecone_apikey,
    #     environment=pinecone_environment
    # )

    index_name = "anantha"

    index = PineconeLangchain.from_existing_index(index_name, embeddings)
    return index


def similar_docs(query, k,pinecone_index_name, embeddings, unique_id):

    index_name = pinecone_index_name

    index = pull_to_pinecone(embeddings)
    similar_docs1 = index.similarity_search_with_score(query, int(k), {"unique_id": unique_id})

    return similar_docs1


def summary():
    llm = OpenAI(temperature=0)



