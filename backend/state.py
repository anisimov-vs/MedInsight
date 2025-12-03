from typing import Annotated, Optional, Dict, Any
from langgraph.graph.message import add_messages
from langgraph.prebuilt.chat_agent_executor import AgentState

class MedicalAgentState(AgentState):
    """Extended agent state with visualization support."""
    visualization_json: Optional[Dict[str, Any]] = None
    final_response: Optional[Dict[str, Any]] = None
