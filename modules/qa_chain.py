from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate

def create_qa_chain(vectorstore):

    pipe = pipeline(
        "text-generation",
        model="tiiuae/falcon-rw-1b",
        max_new_tokens=200
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template = """
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer in one or two sentences.
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context","question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
        chain_type_kwargs={"prompt":PROMPT}
    )

    return qa_chain