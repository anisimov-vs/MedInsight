# Руководство по созданию Tools

## Структура проекта

```
backend/
├── tools.py      # Все инструменты агента
├── graph.py      # Граф LangGraph
├── state.py      # Состояние агента
├── database.py   # Работа с БД
└── api.py        # FastAPI endpoints
```

## Как добавить новый Tool

### 1. Базовый инструмент (возвращает строку)

```python
# backend/tools.py

from langchain_core.tools import tool

@tool("my_tool_name")
def my_tool(param1: str, param2: int) -> str:
    """
    Описание инструмента для LLM.
    Это описание LLM использует чтобы понять когда вызывать инструмент.
    """
    # Логика инструмента
    result = do_something(param1, param2)
    return f"Результат: {result}"
```

### 2. Инструмент с Pydantic схемой (для сложных параметров)

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class MyToolInput(BaseModel):
    table: Literal["patients", "drugs"] = Field(description="Таблица для поиска")
    keywords: List[str] = Field(description="Ключевые слова")

@tool("my_tool", args_schema=MyToolInput)
def my_tool(table: str, keywords: List[str]) -> str:
    """Поиск по таблице."""
    # ...
    return result
```

### 3. Инструмент с Command (для обновления состояния)

Используйте `Command` когда инструмент должен:
- Обновить состояние агента (например, сохранить визуализацию)
- Вернуть сообщение в историю

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command

@tool("save_result")
def save_result(
    data: str,
    tool_call_id: Annotated[str, InjectedToolCallId]  # Автоматически инжектится
) -> Command:
    """Сохраняет результат."""
    
    return Command(update={
        # Обновление состояния
        "my_custom_field": data,
        # Сообщение в историю
        "messages": [ToolMessage("Результат сохранён", tool_call_id=tool_call_id)]
    })
```

### 4. Инструмент с доступом к состоянию

```python
from langgraph.prebuilt import InjectedState

@tool("context_aware_tool")
def context_aware_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]  # Текущее состояние агента
) -> Command:
    """Инструмент с доступом к состоянию."""
    
    # Доступ к предыдущим сообщениям
    messages = state.get("messages", [])
    
    # Доступ к кастомным полям состояния
    viz = state.get("visualization_json")
    
    return Command(update={
        "messages": [ToolMessage("Done", tool_call_id=tool_call_id)]
    })
```

## Регистрация инструмента

После создания инструмента добавьте его в список возвращаемых инструментов:

```python
# backend/tools.py

def create_tools(db: Database):
    
    @tool("search_codes")
    def search_codes(...): ...
    
    @tool("execute_sql")
    def execute_sql(...): ...
    
    @tool("my_new_tool")  # Новый инструмент
    def my_new_tool(...): ...
    
    # Добавить в список
    return [search_codes, execute_sql, my_new_tool]
```

## Добавление поля в состояние

Если инструмент должен сохранять данные в состояние:

```python
# backend/state.py

from typing import Optional, Dict, Any
from langgraph.prebuilt.chat_agent_executor import AgentState

class MedicalAgentState(AgentState):
    """Состояние агента."""
    visualization_json: Optional[Dict[str, Any]] = None
    final_response: Optional[Dict[str, Any]] = None
    my_new_field: Optional[str] = None  # Новое поле
```

## Пример: Инструмент для экспорта в CSV

```python
@tool("export_to_csv")
def export_to_csv(
    sql: Annotated[str, "SQL запрос для экспорта"],
    filename: Annotated[str, "Имя файла"],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Экспортирует результат SQL запроса в CSV файл."""
    try:
        df, err = db.execute(sql)
        if err:
            return Command(update={
                "messages": [ToolMessage(f"Ошибка: {err}", tool_call_id=tool_call_id)]
            })
        
        path = f"/exports/{filename}.csv"
        df.to_csv(path, index=False)
        
        return Command(update={
            "messages": [ToolMessage(f"Файл сохранён: {path}", tool_call_id=tool_call_id)]
        })
    except Exception as e:
        return Command(update={
            "messages": [ToolMessage(f"Ошибка: {e}", tool_call_id=tool_call_id)]
        })
```

## Советы

1. **Описание важно** — LLM решает какой инструмент вызвать на основе описания
2. **Обработка ошибок** — всегда возвращайте понятное сообщение об ошибке
3. **Логирование** — используйте `log("Tool", "message", "C")` для отладки
4. **Превью данных** — не возвращайте огромные данные, делайте превью
5. **Command vs строка** — используйте Command только когда нужно обновить состояние
