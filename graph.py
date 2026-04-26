import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pinecone import Pinecone

load_dotenv()

class AgentState(TypedDict):
    user_message: str
    chat_history: list
    category: str
    response: str

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_retriever(namespace: str):
    """
    Creates a Pinecone retriever scoped to a specific namespace.

    Args:
        namespace: One of 'shipping', 'returns', 'billing', 'account'

    Returns:
        A LangChain retriever that searches only that namespace.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG TOOL 

def create_rag_tool(namespace: str):
    """
    Creates a RAG search tool scoped to a specific Pinecone namespace.

    Args:
        namespace: The namespace this tool will search in.

    Returns:
        A LangChain tool function.
    """

    def search_fn(query: str) -> str:
        try:
            retriever = get_retriever(namespace)
            docs = retriever.invoke(query)

            if not docs:
                return f"No relevant information found in {namespace} knowledge base."

            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "unknown")
                results.append(f"[Source {i}: {source}]\n{doc.page_content}")

            return "\n\n---\n\n".join(results)

        except Exception as e:
            return f"Error searching {namespace} knowledge base: {str(e)}"

    search_fn.__name__ = f"search_{namespace}_knowledge_base"
    search_fn.__doc__ = (
        f"Search the {namespace} knowledge base to find accurate answers "
        f"about customer {namespace} related questions. "
        f"Always use this tool before answering any {namespace} question."
    )

    return tool(search_fn)


# SPECIALIZED AGENTS

def create_specialized_agent(namespace: str, system_prompt: str) -> AgentExecutor:
    """
    Creates a specialized LangChain agent for a specific department.

    Args:
        namespace: Pinecone namespace this agent searches.
        system_prompt: The agent's specific instructions and personality.

    Returns:
        An AgentExecutor ready to handle queries.
    """

    rag_tool = create_rag_tool(namespace)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=[rag_tool],
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=[rag_tool],
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True
    )

# SYSTEM PROMPTS (one per specialized agent)

SHIPPING_PROMPT = """You are a Shipping Support Specialist for ShopEasy.
You ONLY handle questions about shipping, delivery, tracking, and logistics.
Always search the knowledge base before answering.
Be empathetic if a customer's package is missing or delayed.
If the issue requires a ticket or human escalation, tell the customer to 
describe their issue fully so a ticket can be raised.
Keep responses clear, helpful, and concise."""

RETURNS_PROMPT = """You are a Returns and Refunds Specialist for ShopEasy.
You ONLY handle questions about returns, refunds, damaged items, and exchanges.
Always search the knowledge base before answering.
Be understanding — customers requesting refunds are often frustrated.
Walk customers through the return process step by step when needed.
Keep responses clear, helpful, and concise."""

BILLING_PROMPT = """You are a Billing Support Specialist for ShopEasy.
You ONLY handle questions about payments, invoices, discount codes, and charges.
Always search the knowledge base before answering.
Be reassuring when customers have payment concerns or unexpected charges.
Never ask for full card numbers — only last 4 digits if needed for reference.
Keep responses clear, helpful, and concise."""

ACCOUNT_PROMPT = """You are an Account Support Specialist for ShopEasy.
You ONLY handle questions about account access, passwords, loyalty points,
profile settings, and email preferences.
Always search the knowledge base before answering.
Be careful with security-related issues — verify intent before guiding changes.
Keep responses clear, helpful, and concise."""


# INITIALIZE ALL SPECIALIZED AGENTS

shipping_agent = create_specialized_agent("shipping", SHIPPING_PROMPT)
returns_agent = create_specialized_agent("returns", RETURNS_PROMPT)
billing_agent = create_specialized_agent("billing", BILLING_PROMPT)
account_agent = create_specialized_agent("account", ACCOUNT_PROMPT)


# LANGGRAPH NODES

def supervisor_node(state: AgentState) -> dict:
    """
    SUPERVISOR NODE — the router of the graph.

    Reads the user's message and classifies it into one of 4 categories.
    The category determines which specialized agent runs next.

    This uses a simple LLM call — no tools needed here.
    The supervisor's only job is to CLASSIFY, not to answer.
    """

    classification_prompt = f"""You are a customer support routing system for ShopEasy.
    
Classify the following customer message into EXACTLY one of these categories:
- shipping   → questions about delivery, tracking, shipment status, delays
- returns    → questions about returns, refunds, damaged items, exchanges  
- billing    → questions about payments, invoices, charges, discount codes
- account    → questions about login, password, loyalty points, profile settings

Customer message: "{state['user_message']}"

Respond with ONLY the category word. Nothing else. No punctuation."""

    result = llm.invoke(classification_prompt)

    category = result.content.strip().lower()

    valid_categories = ["shipping", "returns", "billing", "account"]
    if category not in valid_categories:
        category = "shipping"

    print(f"[Supervisor] Routed to: {category}")

    return {"category": category}


def shipping_node(state: AgentState) -> dict:
    """
    SHIPPING NODE — runs the shipping specialized agent.
    Reads user_message and chat_history from state.
    Writes the agent's response back to state.
    """

    history = []
    for msg in state["chat_history"]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    result = shipping_agent.invoke({
        "input": state["user_message"],
        "chat_history": history
    })

    return {"response": result["output"]}


def returns_node(state: AgentState) -> dict:
    """
    RETURNS NODE — runs the returns specialized agent.
    Same pattern as shipping_node.
    """

    history = []
    for msg in state["chat_history"]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    result = returns_agent.invoke({
        "input": state["user_message"],
        "chat_history": history
    })

    return {"response": result["output"]}


def billing_node(state: AgentState) -> dict:
    """
    BILLING NODE — runs the billing specialized agent.
    Same pattern as shipping_node.
    """

    history = []
    for msg in state["chat_history"]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    result = billing_agent.invoke({
        "input": state["user_message"],
        "chat_history": history
    })

    return {"response": result["output"]}


def account_node(state: AgentState) -> dict:
    """
    ACCOUNT NODE — runs the account specialized agent.
    Same pattern as shipping_node.
    """

    history = []
    for msg in state["chat_history"]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    result = account_agent.invoke({
        "input": state["user_message"],
        "chat_history": history
    })

    return {"response": result["output"]}

# ROUTING FUNCTION

def route_to_agent(state: AgentState) -> Literal["shipping", "returns", "billing", "account"]:
    """
    Reads the category set by the supervisor and returns
    the name of the next node to execute.

    LangGraph uses this return value to decide which node runs next.
    """
    return state["category"]


# BUILD LANGGRAPH GRAPH

def build_graph():
    """
    Builds and compiles the LangGraph multi-agent graph.

    Returns:
        A compiled LangGraph application ready to invoke.
    """

    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("shipping", shipping_node)
    graph.add_node("returns", returns_node)
    graph.add_node("billing", billing_node)
    graph.add_node("account", account_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",           
        route_to_agent,         
        {                       
            "shipping": "shipping",
            "returns": "returns",
            "billing": "billing",
            "account": "account",
        }
    )

    graph.add_edge("shipping", END)
    graph.add_edge("returns", END)
    graph.add_edge("billing", END)
    graph.add_edge("account", END)

    return graph.compile()

# COMPILED GRAPH 

support_graph = build_graph()

def run_graph(user_message: str, chat_history: list) -> dict:
    """
    Runs the full multi-agent graph for a user message.

    Args:
        user_message: The customer's latest message.
        chat_history: Previous conversation as list of dicts.

    Returns:
        Dict with 'response' (answer) and 'category' (which agent handled it).
    """
    initial_state = AgentState(
        user_message=user_message,
        chat_history=chat_history,
        category="",       
        response=""        
    )

    try:
        final_state = support_graph.invoke(initial_state)

        return {
            "response": final_state["response"],
            "category": final_state["category"]
        }

    except Exception as e:
        return {
            "response": f"I'm sorry, I encountered an error. Please try again. (Error: {str(e)})",
            "category": "unknown"
        }
