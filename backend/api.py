import json
import asyncio
import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage

from backend.graph import medical_graph
from backend.config import cfg, log

app = FastAPI(title="Medical Insight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None


class ConversationHistoryRequest(BaseModel):
    thread_id: str


def extract_answer(resp) -> str:
    """Extract answer text from various response formats."""
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except:
            return resp
    if isinstance(resp, dict):
        return resp.get('answer', str(resp))
    return str(resp)


def get_chart_title(viz) -> str:
    """Extract title from visualization."""
    if viz and isinstance(viz, dict):
        layout = viz.get('layout', {})
        title = layout.get('title', {})
        if isinstance(title, dict):
            return title.get('text', '')
        return str(title) if title else ''
    return ''


async def graph_event_stream(query: str, thread_id: str = None):
    """Stream graph execution events."""
    if not thread_id:
        thread_id = f"session_{uuid.uuid4()}"
        log("API", f"New conversation: {thread_id}", "G")
    else:
        log("API", f"Continuing: {thread_id}", "C")

    config = {
        "recursion_limit": cfg.MAX_TOOL_CALLS * 2 + 1,
        "configurable": {"thread_id": thread_id}
    }

    inputs = {"messages": [HumanMessage(content=query)]}
    step_count = 0
    last_event_time = time.time()
    last_viz = None
    final_sent = False
    last_text = ""

    try:
        async for event in medical_graph.graph.astream(inputs, stream_mode="updates", config=config):
            now = time.time()
            duration = now - last_event_time
            last_event_time = now

            for node_name, node_val in event.items():
                # Handle visualization
                if "visualization_json" in node_val:
                    viz = node_val["visualization_json"]
                    if viz and viz != last_viz:
                        last_viz = viz
                        yield f"data: {json.dumps({'type': 'visualization', 'data': viz})}\n\n"

                # Handle final response
                if "final_response" in node_val and node_val["final_response"]:
                    resp = node_val["final_response"]
                    answer = extract_answer(resp)
                    insights = resp.get('insights', []) if isinstance(resp, dict) else []
                    yield f"data: {json.dumps({'type': 'final', 'answer': answer, 'insights': insights, 'visualization': last_viz, 'thread_id': thread_id})}\n\n"
                    final_sent = True

                # Handle messages
                if "messages" in node_val:
                    for msg in node_val["messages"]:
                        if isinstance(msg, AIMessage):
                            step_count += 1
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    if tc["name"] != "final_answer":
                                        yield f"data: {json.dumps({'type': 'step', 'step': step_count, 'tool': tc['name'], 'duration': duration})}\n\n"
                            elif msg.content:
                                content = msg.content
                                if isinstance(content, list):
                                    content = " ".join(str(c.get('text', c)) if isinstance(c, dict) else str(c) for c in content)
                                last_text = str(content)
                                yield f"data: {json.dumps({'type': 'step', 'step': step_count, 'tool': 'thought', 'thought': last_text[:500], 'duration': duration})}\n\n"
                        elif hasattr(msg, 'content') and msg.content and msg.content != "Answer submitted.":
                            preview = str(msg.content)[:300] + "..." if len(str(msg.content)) > 300 else str(msg.content)
                            yield f"data: {json.dumps({'type': 'tool_result', 'result': preview, 'duration': duration})}\n\n"

            await asyncio.sleep(0.01)

        # If no final was sent, create one
        if not final_sent:
            # Use last_text, or chart title, or default
            answer = last_text or get_chart_title(last_viz) or 'График построен'
            yield f"data: {json.dumps({'type': 'final', 'answer': answer, 'insights': [], 'visualization': last_viz, 'thread_id': thread_id})}\n\n"

    except Exception as e:
        log("API", f"Stream error: {e}", "R")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: QueryRequest):
    return StreamingResponse(
        graph_event_stream(req.query, req.thread_id),
        media_type="text/event-stream"
    )


@app.post("/chat/history")
async def get_conversation_history(req: ConversationHistoryRequest):
    try:
        config = {"configurable": {"thread_id": req.thread_id}}
        state = medical_graph.graph.get_state(config)

        if not state or not state.values:
            return {"messages": [], "thread_id": req.thread_id}

        messages = [
            {"role": msg.type, "content": msg.content}
            for msg in state.values.get("messages", [])
            if hasattr(msg, 'type') and msg.type in ['human', 'ai']
        ]
        return {"messages": messages, "thread_id": req.thread_id}
    except Exception as e:
        log("API", f"History error: {e}", "R")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/history/{thread_id}")
async def delete_conversation(thread_id: str):
    log("API", f"Deleted: {thread_id}", "Y")
    return {"status": "deleted", "thread_id": thread_id}


@app.get("/health")
def health():
    return {"status": "ok", "checkpointer": "memory"}


@app.get("/")
def root():
    return {"service": "Medical Insight API", "version": "3.0.0"}
