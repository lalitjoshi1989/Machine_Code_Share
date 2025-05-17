from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.2,
            max_tokens=512,
        )