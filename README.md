# Genai_chatbot
📄 Sai Charan’s PDF ChatBot (RAG using Groq + LangChain)

This project is a Streamlit-based AI chatbot that allows users to upload a PDF document and ask questions about its content.
The chatbot uses Retrieval-Augmented Generation (RAG) powered by Groq LLMs, LangChain, FAISS, and HuggingFace embeddings.

🚀 Features

📂 Upload any PDF document

🔍 Intelligent document chunking

🧠 Semantic search using FAISS vector database

🤖 Large Language Model powered by Groq (LLaMA 3.3 – 70B)

💬 Ask questions and get answers only from the PDF content

⚡ Fast and interactive UI using Streamlit

🧱 Tech Stack

Frontend: Streamlit

LLM: Groq (llama-3.3-70b-versatile)

Framework: LangChain

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Vector Store: FAISS

PDF Parsing: PyPDF2

Environment Management: python-dotenv

Project Structure :
.

├── test.py                # Main Streamlit app

├── .env                   # Environment 

├── requirements.txt       # Dependencies

└── README.md              # Project documentation
