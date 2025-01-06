from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from flask_sqlalchemy import SQLAlchemy
import os
import pandas as pd

app = Flask(__name__)

# Configure Neon PostgreSQL Database (replace with your actual Neon URL)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://ChatBotLog_owner:01YZFuoOgsBK@ep-billowing-night-a5om12fl.us-east-2.aws.neon.tech/ChatBotLog?sslmode=require'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define a model to store questions and responses
class QAHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)

    def __init__(self, question, response):
        self.question = question
        self.response = response

# Initialize the database
with app.app_context():
    db.create_all()

# Load environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyDzLtb6wqyogXc8DMCjJWVhvi1o8cIkrvM"  # Replace with your actual API key

# Load documents and setup embeddings
faculty = pd.read_csv("satic/Faculty_details - faculty_datanew.csv")
faculty_str = faculty.to_string()
pdf_path = 'satic/Alwin_merged_merged_merged.pdf'  # Ensure this path is correct and the PDF exists here
loader = PyPDFLoader(pdf_path)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, max_tokens=500, google_api_key=os.getenv("GOOGLE_API_KEY"))

system_prompt = (
    '''College is accredited with naac "A" Grade.'''
     "You are a friendly and knowledgeable assistant designed to help with inquiries about the college for Adi Shankara Institute of Engineering and Technology. Provide clear and accurate answers based on the   provided context. If exact information isn't available, respond positively, offering alternatives or highlighting related facilities. For example, if the college lacks a certain amenity, such as a swimming pool, emphasize other available facilities or services. Keep responses concise"
    "\n\n"
    "Context: {context}"
    "cse == cs, that is it is same department. Do not give separate answers when asked about both"
    f"if about faculty details is asked answer from {faculty_str}. Also give the link of the faculty to get more info about the faculty from view profile column.If hod is asked look for head of departments. "
    "Only Answer from the given document, do not answer from outside of its knowledge. "
    "Asiet means Adi Shankara Institute of Engineering and Technology."
    "Beautify with spaces and start in new line when necessary."
    "cse and cse(ai) are separate departments."
    "cse(ai) and cse(ds) are same department."
    
    "ug cousrses"
    '''BTECH OR UG ADMISSION PROCEDURE:For both management and government quotas, candidates must be Indian citizens and at least 17 years old  (no exemptions). Applicants must have passed the Higher Secondary Examination of the Kerala Board or an equivalent exam with a minimum of 45% aggregate in Physics, Chemistry, and Mathematics (PCM), without rounding off marks. Additionally, candidates must qualify in the Engineering Entrance Exam conducted by the Commissioner of Entrance Exams, Kerala.'''
    
    "ASIET have achieved 5th rank in the Kerala institutional ranking framework (KIRF) 2024 among all self financing colleges"
    f"If anyone asks about a faculty member's publications or research area, provide the information from {faculty_str}. Additionally, include the profile link for more details. The profile link can be found in {faculty_str}."
    "Adi Shankara Institute of Engineering and Technology (ASIET) offers the following courses: Civil Engineering, Computer Science & Engineering, Computer Science & Engineering (Artificial Intelligence), Computer Science & Engineering (Data Science), Electronics & Biomedical Engineering, Electronics & Communication Engineering, Electrical & Electronics Engineering, Mechanical Engineering, and Robotics and Automation,MCA,MTech"
    "Beautify the output"
    ""
    '''Give any link in <a></a> format
    for example: "Gayathri Dili is an Assistant Professor in the Computer Science and Engineering (Artificial Intelligence) department at ASIET. You can find her profile at <a href='https://www.adishankara.ac.in/department/faculty/314'>Profile</a> for more information."
        '''
    '''If a question is asked about any person, search for their name in the faculty list and provide their profile link if available. For any other topic, include a relevant link, ensuring that every URL is enclosed within an <a></a> tag.'''
    "give outputs in html elements and beautify ouput"
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
def ask():
    question = request.form['question']
    response = rag_chain.invoke({"input": question})
    answer = response['answer']  # Extract the answer from the response object

    # Save the question and answer to the database
    new_qa = QAHistory(question=question, response=answer)
    db.session.add(new_qa)
    db.session.commit()

    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
