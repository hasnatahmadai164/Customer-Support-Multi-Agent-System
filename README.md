## Customer Support Multi Agent System

A multi-agent AI customer support system for an e-commerce platform. A supervisor agent classifies incoming customer queries and routes them to one of four specialized agents, each backed by its own PDF knowledge base stored in Pinecone.

## Agent Capabilities

**Supervisor Agent**
Reads the customer message and classifies it into one of four categories — shipping, returns, billing, or account. Does not answer questions. Only routes to the correct specialized agent.

**Shipping Agent**
Handles questions about delivery timelines, shipping options, order tracking, lost packages and international shipping.

**Returns Agent**
Handles questions about return eligibility, how to initiate a return, damaged or defective items and refund processing timelines.

**Billing Agent**
Handles questions about accepted payment methods, failed payments, invoices, duplicate charges and discount codes.

**Account Agent**
Handles questions about password reset, account lockouts, two-factor authentication, loyalty points and account settings.

## Tech Stack

- LangGraph   (Multi-agent workflow)
- LangChain   (Agent and RAG framework)
- OpenAI GPT-4o-mini  (LLM)
- Pinecone   (Vector database with separate namespace per agent)
- FastAPI   (Backend API)
- HTML, CSS, JavaScript   (Chat frontend)

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Add your credentials to the `.env` file

3. Ingest PDF documents into Pinecone
```bash
python knowledge_base.py
```

4. Start the server
```bash
uvicorn main:app --reload
```

5. Open `http://localhost:8000` in your browser

## Environment Variables

```
OPENAI_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_CLOUD
PINECONE_REGION
```
