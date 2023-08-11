# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader

# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# set API key for OpenAI
# (sub out here for other LLMs)
os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0, verbose=True)
embeddings = OpenAIEmbeddings()

def get_answer_pdf(file, query) -> str:
    # create and load PDF loader
    loader = PyPDFLoader(file.getvalue())

    # split pages from pdf
    pages = loader.load_and_split()
    # load documents into vector database aka chromaDB
    store = Chroma.from_documents(pages, embeddings, collection_name='brochure')

    # create vectorstore info object - metadata repo
    vectorstore_info = VectorStoreInfo(
        name="brochure",
        description="a brochure for an esmt program",
        vectorstore=store
    )
    # convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    # pass the query to the LLM
    # response = llm(query)
    # swap out the raw llm for a document agent
    response = agent_executor.run(query)
    # ..and write out to the screen
    st.write(response)

    # with a streamlit expander
    with st.expander('Document Similarity Search'):
        # find relevant pages
        search = store.similarity_search_with_score(query)
        # write out the first
        st.write(search[0][0].page_content)

st.header("PDF Bot v2")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
        query = st.text_area("Ask me a question related about the document...")
        button = st.button("Submit")
        if button:
            st.write(get_answer_pdf(uploaded_file, query))