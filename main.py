import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from graph import run_graph

load_dotenv()

# FASTAPI APP INITIALIZATION
app = FastAPI(
    title="ShopEasy Multi-Agent Support API",
    description="Multi-agent customer support powered by LangGraph and OpenAI",
    version="2.0.0"
)

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST MODELS (Pydantic)
class ChatRequest(BaseModel):
    """
    Structure of the request body the frontend sends.
    user_message: the customer's latest message
    chat_history: previous conversation turns for memory
    """
    user_message: str
    chat_history: list = []  


class ChatResponse(BaseModel):
    """
    Structure of the response we send back to the frontend.
    response: the agent's answer
    category: which specialized agent handled it (for display)
    """
    response: str
    category: str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the frontend HTML page.
    When you open http://localhost:8000 in your browser,
    this function runs and returns the HTML content.
    """
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Receives user message and returns AI response.

    FastAPI automatically:
    - Parses the JSON request body into ChatRequest object
    - Validates the data types
    - Returns ChatResponse as JSON to the frontend
    """

    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    result = run_graph(
        user_message=request.user_message,
        chat_history=request.chat_history
    )

    return ChatResponse(
        response=result["response"],
        category=result["category"]
    )

@app.get("/health")
async def health_check():
    """
    Simple health check. Returns {"status": "ok"} if server is running.
    Visit http://localhost:8000/health to verify the server is up.
    """
    return {"status": "ok", "message": "ShopEasy Support API is running"}
