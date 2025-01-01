from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

app = Flask(__name__)

# Load environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyDzLtb6wqyogXc8DMCjJWVhvi1o8cIkrvM"  # Replace with your actual API key or use environment setup

# Load documents and setup embeddings
pdf_path = 'satic/chst.pdf'  # Ensure this path is correct and the PDF exists here
loader = PyPDFLoader(pdf_path)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500, google_api_key=os.getenv("GOOGLE_API_KEY"))

system_prompt = (
    "You are a friendly and knowledgeable assistant designed to help with inquiries about the college. Provide clear and accurate answers based on the provided context. If exact information isn't available, respond positively, offering alternatives or highlighting related facilities. For example, if the college lacks a certain amenity, such as a swimming pool, emphasize other available facilities or services. Keep responses concise, with up to three sentences."
    "\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    response = rag_chain.invoke({"input": question})
    # Extract the actual answer from the response object
    answer = response['answer']
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(host='13.201.78.220', debug=True)
