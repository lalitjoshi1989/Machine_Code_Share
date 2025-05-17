from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from load_data_models.load_files import download_embed_model, load_pdf_file, text_split
from load_data_models.prompt import system_prompt, llm
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

app = Flask(__name__)
app.secret_key = "supersecret"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
embeddings = download_embed_model()

llm = llm
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf = request.files["pdf"]
        if pdf.filename == "":
            return "No file selected"

        file_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
        pdf.save(file_path)

        documents = load_pdf_file(UPLOAD_FOLDER)
        chunks = text_split(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")

        session["pdf_uploaded"] = True
        return redirect(url_for("answer"))

    return render_template("index.html")


@app.route("/answer", methods=["GET", "POST"])
def answer():
    response = None

    if not session.get("pdf_uploaded"):
        return redirect(url_for("index"))

    if request.method == "POST":
        question = request.form["question"]
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input":question})
        response_final = str(response['answer'])
        return render_template("answer2.html", answer=response_final)
    else:
        return render_template("answer2.html")

if __name__ == "__main__":
    app.run(debug=True, port = 1181)