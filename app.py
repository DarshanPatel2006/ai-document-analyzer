import streamlit as st

from modules.loader import load_documents
from modules.splitter import split_documents
from modules.embeddings import create_vector_store
from modules.qa_chain import create_qa_chain

st.title("AI Document Assistant")

uploaded_file = st.file_uploader("Upload PDF")

if uploaded_file:

    with open("data/sample.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = load_documents("data/sample.pdf")
    chunks = split_documents(docs)

    vectorstore = create_vector_store(chunks)

    qa_chain = create_qa_chain(vectorstore)

    question = st.text_input("Ask a question")

    if question:

        # convert all document text
        full_text = " ".join([doc.page_content for doc in docs])

        # check counting question
        if "how many" in question.lower() and "times" in question.lower():

            if '"' in question:
                word = question.split('"')[1]
            else:
                word = question.split()[-1]

            count = full_text.lower().count(word.lower())

            st.write(f'The word "{word}" appears {count} times in the document.')

        else:

            result = qa_chain({"query": question})
            st.write(result["result"])