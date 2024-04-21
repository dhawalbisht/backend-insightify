from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask_cors import CORS
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

global_vectorstore = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def generate_summary(text_chunks):
    llm = ChatOpenAI()
    summary_prompt = "Provide a comprehensive summary of the all the text. The summary should cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format. The summary should be no more than a 150 words:\n\n"
    summary_prompt += "\n\n".join(text_chunks)
    summary = llm.call_as_llm(summary_prompt)
    return summary

@app.route('/process_pdfs', methods=['POST'])
def process_pdfs():
    global global_vectorstore
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    global_vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(global_vectorstore)
    summary = generate_summary(text_chunks)
    return jsonify({'summary': summary, 'conversation_chain': conversation_chain.memory.chat_memory.messages})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    global global_vectorstore
    question = request.json['question']
    chat_history = request.json['chat_history']
    
    # Convert chat_history to the expected format
    formatted_chat_history = []
    for message in chat_history:
        formatted_chat_history.append((message['role'], message['content']))
    
    conversation_chain = get_conversation_chain(global_vectorstore)
    response = conversation_chain({'question': question, 'chat_history': formatted_chat_history})
    
    # Extract the answer and updated chat history from the response
    answer = response['answer']
    updated_chat_history = response['chat_history']
    
    # Convert the updated chat history to a serializable format
    serializable_chat_history = []
    for message in updated_chat_history:
        if isinstance(message, tuple):
            serializable_chat_history.append({
                'role': message[0],
                'content': message[1]
            })
        else:
            serializable_chat_history.append({
                'role': message.type,
                'content': message.content
            })
    
    return jsonify({'answer': answer, 'chat_history': serializable_chat_history})



if __name__ == '__main__':
    app.run()