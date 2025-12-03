import json
import traceback
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from backend.database import Database
from backend.config import log


class SearchCodesInput(BaseModel):
    table: Literal["diagnoses", "drugs"] = Field(description="Table to search")
    keywords: List[str] = Field(description="Medical terms (Russian/English) or ICD codes")


class FinalAnswerInput(BaseModel):
    answer: str = Field(description="Final text answer in Russian")
    insights: List[str] = Field(default_factory=list, description="Key insights")


def create_tools(db: Database):
    @tool("search_codes", args_schema=SearchCodesInput)
    def search_codes(table: str, keywords: List[str]) -> str:
        """Search for diagnosis or drug codes using FTS and fuzzy matching."""
        try:
            log("Tool", f"search_codes: {table}, {keywords}", "C")
            df_desc, err = db.execute(f"DESCRIBE {table}")
            if err:
                return f"DB Error: {err}"

            cols = df_desc['column_name'].tolist()
            col_id = "diagnosis_code" if "diagnosis_code" in cols else "drug_code"
            col_text = "diagnosis_name" if "diagnosis_name" in cols else "full_name"

            valid_kw = [k for k in keywords if len(k) >= 1]
            if not valid_kw:
                return "Error: Keywords too short."

            results = []
            search_q = " ".join(valid_kw).replace("'", "")

            # FTS search
            sql_fts = f"""
                SELECT {col_id}, {col_text}, fts_main_{table}.match_bm25({col_id}, ?) AS score
                FROM {table} WHERE score IS NOT NULL ORDER BY score DESC LIMIT 20
            """
            df_fts, _ = db.execute(sql_fts, [search_q])
            if df_fts is not None and not df_fts.empty:
                results.append(f"FTS Matches:\n{df_fts.to_string(index=False)}")

            # Fuzzy search
            conditions, params = [], []
            for kw in valid_kw:
                sub = [f"{col_text} ILIKE ?", f"{col_id} ILIKE ?"]
                params.extend([f"%{kw}%", f"{kw}%"])
                if len(kw) > 3:
                    sub.append(f"levenshtein({col_text}, ?) <= 3")
                    params.append(kw)
                conditions.append(f"({' OR '.join(sub)})")

            sql_fuzzy = f"SELECT {col_id}, {col_text} FROM {table} WHERE {' OR '.join(conditions)} LIMIT 20"
            df_fuzzy, _ = db.execute(sql_fuzzy, params)
            if df_fuzzy is not None and not df_fuzzy.empty:
                results.append(f"Fuzzy Matches:\n{df_fuzzy.to_string(index=False)}")

            return "\n\n".join(results) if results else "No records found."
        except Exception:
            return f"Error: {traceback.format_exc()}"

    @tool("execute_sql")
    def execute_sql(
        sql: Annotated[str, "Valid DuckDB SQL query"],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Execute SQL and return preview."""
        try:
            log("Tool", f"execute_sql: {sql[:150]}...", "C")
            df, err = db.execute(sql)
            if err:
                return Command(update={"messages": [ToolMessage(f"SQL Error: {err}", tool_call_id=tool_call_id)]})
            if df is None or df.empty:
                return Command(update={"messages": [ToolMessage("Empty result", tool_call_id=tool_call_id)]})

            preview = f"Rows: {len(df)}. Cols: {list(df.columns)}\n{df.head(10).to_string(index=False)}"
            return Command(update={"messages": [ToolMessage(preview, tool_call_id=tool_call_id)]})
        except Exception:
            return Command(update={"messages": [ToolMessage(f"Error: {traceback.format_exc()}", tool_call_id=tool_call_id)]})

    @tool("generate_visualization")
    def generate_visualization(
        sql: Annotated[str, "SQL query for data"],
        python_code: Annotated[str, "Python code creating Plotly fig. df=query results"],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Generate Plotly chart from SQL data."""
        try:
            log("Tool", "generate_visualization", "C")
            df, err = db.execute(sql)
            if err:
                return Command(update={"messages": [ToolMessage(f"SQL Error: {err}", tool_call_id=tool_call_id)]})
            if df is None or df.empty:
                return Command(update={"messages": [ToolMessage("No data for visualization", tool_call_id=tool_call_id)]})

            # Clean code
            code = python_code.strip()
            for prefix in ["```python\n", "```\n", "python\n"]:
                if code.startswith(prefix):
                    code = code[len(prefix):]
            code = code.rstrip("`").strip()
            code = "\n".join(l for l in code.split("\n") if not l.strip().startswith("import "))

            exec_globals = {
                "df": df, "px": px, "go": go, "pd": pd, "np": np,
                "__builtins__": {
                    "len": len, "str": str, "int": int, "float": float, "bool": bool,
                    "list": list, "dict": dict, "range": range, "enumerate": enumerate,
                    "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
                    "sorted": sorted, "zip": zip, "True": True, "False": False, "None": None
                }
            }
            exec_locals = {}
            exec(code, exec_globals, exec_locals)

            fig = exec_locals.get('fig') or exec_globals.get('fig')
            if not isinstance(fig, go.Figure):
                return Command(update={"messages": [ToolMessage("Code must create 'fig' variable", tool_call_id=tool_call_id)]})

            return Command(update={
                "visualization_json": json.loads(fig.to_json()),
                "messages": [ToolMessage("Visualization created.", tool_call_id=tool_call_id)]
            })
        except Exception:
            return Command(update={"messages": [ToolMessage(f"Error: {traceback.format_exc()}", tool_call_id=tool_call_id)]})

    @tool("final_answer", args_schema=FinalAnswerInput)
    def final_answer(
        answer: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        insights: List[str] = []
    ) -> Command:
        """Submit final response."""
        return Command(update={
            "final_response": {"answer": answer, "insights": insights},
            "messages": [ToolMessage("Answer submitted.", tool_call_id=tool_call_id)]
        })

    return [search_codes, execute_sql, generate_visualization, final_answer]
