import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from backend.database import Database
from backend.tools import create_tools
from backend.state import MedicalAgentState
from backend.config import log, cfg

SYSTEM_PROMPT = """You are an expert Medical Data Analyst working with a Russian medical database.

### CRITICAL RULES:
1. **Always call a tool** - Never output raw text without tool calls
2. **End with `final_answer`** - MUST call final_answer to submit response
3. **Maximum {max_calls} tool calls per turn**

### WORKFLOW:
1. **Discovery**: Use `search_codes` to find ICD-10 diagnosis codes (Russian text)
2. **Analysis**: Use `execute_sql` to query the database (returns first 10 rows)
3. **Visualization**: Use `generate_visualization` with:
   - `sql`: SQL query for data
   - `python_code`: Python code creating `fig` (px, go, pd, np available)
4. **REQUIRED**: Call `final_answer` with your analysis

### Database Schema:
{schema}

### Guidelines:
- Use CTEs (WITH) for complex queries
- Database uses RUSSIAN text for diagnosis names
- Infer context from conversation history for follow-ups
"""


class MedicalGraph:
    def __init__(self):
        self.db = Database("data")
        self.tools = create_tools(self.db)
        self.checkpointer = InMemorySaver()

        # Primary model (Groq)
        self.primary = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            max_retries=0
        ).bind_tools(self.tools)

        # Fallback model (OpenRouter)
        self.fallback = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-oss-20b:free",
            temperature=0
        ).bind_tools(self.tools)

        self.graph = self._build_graph()
        log("Graph", "Initialized with custom StateGraph", "G")

    def _invoke_with_fallback(self, messages):
        """Invoke model with fallback on error."""
        try:
            return self.primary.invoke(messages)
        except Exception as e:
            log("Graph", f"Primary model error: {e}, using fallback", "Y")
            return self.fallback.invoke(messages)

    def _build_graph(self):
        builder = StateGraph(MedicalAgentState)

        def agent(state: MedicalAgentState) -> dict:
            schema = self.db.get_schema()
            system_content = SYSTEM_PROMPT.format(schema=schema, max_calls=cfg.MAX_TOOL_CALLS)
            messages = [{"role": "system", "content": system_content}] + state["messages"]
            
            response = self._invoke_with_fallback(messages)
            
            updates = {"messages": [response]}
            if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
                updates["final_response"] = None
                updates["visualization_json"] = None
            
            return updates

        def route(state: MedicalAgentState) -> str:
            if state.get("final_response"):
                return END
            
            last = state["messages"][-1] if state["messages"] else None
            if isinstance(last, AIMessage) and hasattr(last, 'tool_calls') and last.tool_calls:
                return "tools"
            return END

        builder.add_node("agent", agent)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", route)
        builder.add_edge("tools", "agent")

        return builder.compile(checkpointer=self.checkpointer)


medical_graph = MedicalGraph()
