# 🤖 Agentic Persona-Based RAG Assistant for Technical PDFs

This project is an **agentic AI system** that helps semiconductor industry professionals query technical PDF documents through the lens of different roles. It extends a traditional Retrieval-Augmented Generation (RAG) pipeline by introducing autonomous agent behavior that plans multi-step reasoning tasks using a goal and role-based perspective.

## 🧠 What It Does

- Accepts one or more technical PDF documents
- Embeds and stores document chunks in a FAISS vector index
- Lets users select an industry persona (e.g., IDM, Fabless, Foundry, OEM)
- Users enter a **goal**, and an agent plans steps to fulfill that goal
- Each step retrieves relevant context and generates an LLM-based answer
- Final response is tailored to the persona’s domain-specific priorities

## 👥 Supported Personas

Personas are stored in `personas.txt` and include descriptions like:

- **IDM (Integrated Device Manufacturer)**
- **Fabless Semiconductor Engineer**
- **Capital Equipment Provider**
- **Foundry Representative**
- **Product Manufacturer / OEM**

Each persona adjusts how the system interprets technical content.

## 🧰 Tech Stack

- **Python 3.10+**
- **Streamlit** – User interface
- **OpenAI GPT-4** – Used for both embedding and agent reasoning
- **FAISS** – Vector search index
- **text-embedding-ada-002** – Embedding model for semantic search
- **PyPDF2** – PDF parsing
- **LangChain Agents** – For autonomous, multi-step planning
- **dotenv** – For managing API keys

## 🚀 How to Run It Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/persona-rag-agent.git
   cd persona-rag-agent
