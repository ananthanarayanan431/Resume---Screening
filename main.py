
import streamlit as st
import uuid
from utils import create_docs ,create_embedding_load_data
from utils import push_to_pinecone ,pull_to_pinecone ,similar_docs
from pinecone import Pinecone

import os
from constant import openai,PINECONE_API_KEY

os.environ['OPENAI_API_KEY' ] = openai
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

def main():

    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance..")
    st.subheader("I can you help you Resume Screening process")

    job_description = st.text_area("Please paste the Job Description here" ,key='1')
    document_count = st.text_input("No of resume to return" ,key='2')

    pdf = st.file_uploader("Upload PDF files here" ,type=['pdf'] ,accept_multiple_files=True)

    submit = st.button("Help with Analysis")

    if submit:
        with st.spinner("Wait for it..."):
            st.write("Our process")

            st.session_state['unique_id'] = uuid.uuid4().hex
            st.write(st.session_state['unique_id'])

            docs = create_docs(pdf ,st.session_state['unique_id'])
            # st.write(docs)
            # st.write(len(docs))

            embeddings = create_embedding_load_data()

            api = PINECONE_API_KEY
            env ="gcp-starter"

            index ="anantha"

            push_to_pinecone(index ,embeddings ,docs)

            relevant_docs = similar_docs(job_description ,document_count ,index ,embeddings
                                         ,st.session_state['unique_id'])

            print(relevant_docs)
            st.write(relevant_docs)


            st.success("Hope I was able to save your time")

main()
