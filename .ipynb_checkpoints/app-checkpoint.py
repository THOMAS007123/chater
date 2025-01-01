from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Define the main URL
main_url = 'https://www.adishankara.ac.in/'

# Function to scrape the main URL
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    nav = soup.find('nav')
    nav_urls = []

    if nav:
        links = nav.find_all('a')
        for link in links:
            href = link.get('href')
            if href:
                full_url = href if href.startswith('http') else f"{url.rstrip('/')}/{href.lstrip('/')}"
                nav_urls.append(full_url)
    return nav_urls

# Function to process and retrieve data
def process_data(nav_urls):
    loader = UnstructuredURLLoader(urls=nav_urls)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="YOUR_API_KEY")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    
    return vectorstore

# Load data and prepare retriever
nav_urls = scrape_website(main_url)
vectorstore = process_data(nav_urls)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Create the question-answer chain
system_prompt = (
    "You are a friendly and knowledgeable assistant designed to help with inquiries about the college. "
    "Provide clear and accurate answers based on the provided context. If exact information isn't available, "
    "respond positively, offering alternatives or highlighting related facilities. Keep responses concise."
    "\n\nContext: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500, google_api_key="YOUR_API_KEY")
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask route for the homepage
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form["query"]
        response = rag_chain.invoke({"input": query})
        return render_template("index.html", response=response, query=query)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
