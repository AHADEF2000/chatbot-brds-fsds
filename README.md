📚 Intelligent Document QA Chatbot (BRD/FSD Assistant)

A bilingual (Arabic/English) chatbot for retrieving accurate answers from BRDs, FSDs, and technical documents.

🧩 Project Overview

This chatbot helps users quickly search and understand information inside Business Requirement Documents (BRD) and Functional Specification Documents (FSD).
It answers questions in Arabic and English, retrieves the relevant reference text, and explains complex sections clearly.

The chatbot was originally developed using Python, LangChain, LangGraph, and ChromaDB.
Later, the retrieval pipeline was upgraded to use the OpenAI Assistants Vector Store, which provides:

Built-in document storage

Automatic chunking

High-quality embeddings

Free usage for reasonable datasets

Better retrieval quality than Chroma in many cases

This allowed the chatbot to deliver much more accurate answers, especially in Arabic.

⚙️ Tech Stack
Phase 1 – Local Python RAG

Python (Flask)

LangChain

LangGraph

ChromaDB

OpenAI Embeddings

Custom document chunking

Manual reranking & cosine similarity scoring

Phase 2 – Migration to OpenAI Assistants Vector Store

OpenAI Assistants API

Vector Store + File Storage

Built-in Chunking & Embedding

Automatic Retrieval (no need to manage Chroma locally)

System + Tools + Messages architecture

Tuned:

temperature

top_p

max_output_tokens

This removed the need to pay for external vector databases and improved retrieval accuracy.

🚀 How the Chatbot Works
1. Document Upload

BRDs, FSDs, PDFs, and DOCX files are uploaded to the OpenAI Vector Store, which handles:

Splitting documents into chunks

Generating embeddings

Storing vectors

2. Retrieval

When a user asks a question:

The Assistants API performs semantic search over the vector store

Returns the most relevant chunks

Passes them directly to the model in the context window

This reduces hallucination and improves the factual correctness of answers.

3. Response Generation

Chatbot answers in Arabic or English depending on the user

Adds citations from the retrieved documents

Explains content clearly (BRD → simple explanation, FSD → technical explanation)

4. Fine-tuning Behavior

Inside the OpenAI Assistant:

Temperature is tuned for balanced creativity/accuracy

Top-P is optimized to reduce randomness

Instructions refine tone and style

🧠 Features
✔ AI-powered question answering

Answers questions and extracts relevant sections from BRDs and FSDs.

✔ Bilingual (Arabic + English)

Automatically detects the user’s language and responds accordingly.

✔ High-accuracy retrieval

Using OpenAI’s vector store improves ranking of Arabic text chunks.

✔ Clean and structured responses

Summaries

Step-by-step explanations

Citations from the documents

Follow-up questions supported

✔ Web UI (Flask)

A simple chat interface built using Python/Flask, easy to deploy.

The Chatbot has been deployed using Azure Web service
