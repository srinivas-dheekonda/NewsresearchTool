import os 
import streamlit as st
import pickle
import time 
import langchain
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv ### to load .enve variables 
from groq import Groq
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


# --- CUSTOM CSS ---
def local_css():
    st.markdown(
        """
        <style>
        /* 1. Style the Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #0e1117;
        }

        /* 2. Style the Text Input Borders (Green) */
        /* Targets the div containing the input */
        div[data-baseweb="input"] {
            border: 2px solid #2ecc71 !important;
            border-radius: 8px;
            transition: 0.3s;
        }
        
        /* Highlight border on focus */
        div[data-baseweb="input"]:focus-within {
            border-color: #27ae60 !important;
            box-shadow: 0 0 10px rgba(46, 204, 113, 0.3);
        }

        /* 3. Style the Process Button (Blue) */
        div.stButton > button {
            background-color: #007bff;
            color: white;
            width: 100%;
            border-radius: 20px;
            border: none;
            height: 3em;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        /* Hover effect for the button */
        div.stButton > button:hover {
            background-color: #0056b3;
            border: none;
            color: #ffffff;
            transform: scale(1.02);
        }
        
        /* 4. Sidebar Title Color */
        [data-testid="stSidebarNav"] + div h1 {
            color: #ffffff;
            font-size: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# --- APP LAYOUT ---
st.title("News Research Tool")

st.sidebar.title("News Article URLs")

# Store URLs in a list if you want to use them later
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# --- 4. CUSTOM QUERY BOX ---
# This adds the input box you requested above the answer
query = st.text_input("Question:", placeholder="e.g., What is the price of Tiago iCNG?")

process_url_clicked = st.sidebar.button("Process URLs")

main_placefolder = st.empty()
llm = ChatGroq(
    temperature=0.9, 
    model_name="llama-3.3-70b-versatile",  
    max_tokens=500
)
if process_url_clicked:
    with st.spinner("Analyzing news articles..."):
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        main_placefolder.text("Data Loading....started......")

        ## splitting data 
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n","\n",",","."],
            chunk_size = 1000,
            chunk_overlap = 200
        )
        docs = text_splitter.split_documents(data)

        main_placefolder.text("Text Splitter....started......")

        ## create embedding ans save it to faiss index
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
        ##vectorindex_google = FAISS.from_documents(docs, embeddings)
        ##vectorindex_google.save_local("google_faiss_index")
        ##main_placefolder.text("TSaved the pickle file..........")
        vectorstore = FAISS.load_local("google_faiss_index", embeddings, allow_dangerous_deserialization=True)
    
        def get_car_info(query):
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            return result

        # Your specific query

        ##query = "what is the price of Tiago iCNG?"
        response = get_car_info(query)
        
        st.header("Answer")
        st.write(response)
        

        

        st.success("URLs processed successfully!")


