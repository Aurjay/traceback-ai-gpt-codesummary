import os
import requests
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/api/new-gpt', methods=['POST'])
def my_endpoint():
    try:
        # Download the .py file from the GCP URL
        url = "https://storage.googleapis.com/test-json-latest/test.py"
        file_name = url.split("/")[-1]
        response = requests.get(url)
        response.raise_for_status()

        # Save the file in the current directory
        with open(file_name, "wb") as file:
            file.write(response.content)

        # Read the downloaded .py file contents
        with open(file_name, "r") as file:
            file_contents = file.read()

        # Set the entire file contents as the question
        question = file_contents

        if not question:
            return jsonify({'error': 'Question is missing in the file.'}), 400

        # Get the OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not found.'}), 500

        # Determine the file path
        file_path = 'EU-AI-ACT-2.txt'

        # Load the text documents
        loader = TextLoader(file_path, encoding='utf8')
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Create the embeddings and Chroma index
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(texts, embeddings)

        # Set up the prompt template
        context = "The EU-AI-ACT is a regulatory framework governing AI development, deployment, and use in the European Union. It addresses ethics, transparency, accountability, and data protection. The goal is to ensure AI respects rights, promotes fairness, and manages risks. It includes articles on AI impact assessments, high-risk systems, human oversight, and public administration, fostering a responsible AI ecosystem."
        prompt_template = """Please utilize the context provided to comprehend the question thoroughly and deliver a conversational response. Give a short summary of the input code and explain whether it is compliant with the EU-AI-ACT document or not, providing supporting reasons with references to specific articles. 

        Context:
        {context}

        Question: {question}"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Set up the RetrievalQA chain
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)

        # Run the query
        answer = qa.run(question)
        print("Answer:", answer)

        response = jsonify({'answer': answer})
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/check', methods=['GET'])
def check_endpoint():
    return jsonify({'message': 'API endpoint is working.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
