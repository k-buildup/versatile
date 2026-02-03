from typing import List, Dict, AsyncGenerator
from pydantic import BaseModel, Field
from dataclasses import dataclass
from datetime import datetime
import multiprocessing
from enum import Enum
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.chat_models import ChatLlamaCpp
from llama_cpp import LlamaGrammar
from langchain.tools import tool


# ============================================================================
# Tools
# ============================================================================

class DateTimeInput(BaseModel):
    location: str = Field(description="The city and state, e.g. Seoul, Republic of Korea")

@tool("get_datetime", args_schema=DateTimeInput)
def get_datetime(location: str):
    """Get the current datetime in a given location"""
    return f"Now the datetime in {location} is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


class StockPriceInput(BaseModel):
    symbol: str = Field(description="The stock symbol, e.g. AAPL, MSFT")

@tool("get_stock_price", args_schema=StockPriceInput)
def get_stock_price(symbol: str):
    """Get the current stock price for a given symbol"""
    return f"Now the stock price for {symbol} is $150.00"


# ============================================================================
# Enums and Configs
# ============================================================================

class OutputMode(Enum):
    """출력 모드 정의"""
    BASIC = "basic"
    TODOLIST = "todolist"
    THINK = "think"
    TODO = "todo"
    RESULT = "result"
    LOG = "log"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class ModelConfig:
    """LLM 모델 설정"""
    n_ctx: int = 4096
    n_gpu_layers: int = 20
    n_batch: int = 512
    n_threads: int = 6
    use_mlock: bool = True
    use_mmap: bool = True
    verbose: bool = False


@dataclass
class GenerationConfig:
    """텍스트 생성 설정"""
    max_tokens: int = 8192
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.2


# ============================================================================
# Prompt Builder
# ============================================================================

class PromptBuilder:
    """프롬프트 생성 담당 클래스"""
    
    DEFAULT_SYSTEM_PROMPT = "You are name is Versatile (버사타일).\n\n---\n"
    
    TODOLIST_PROMPT = """
# Important

When you receive user input, break it down into sub-todo lists for step-by-step thinking.
The list should be sorted in logical order,
each sub-todo should be independently answerable,
and the content of sub-todos should not be similar to the user input.

For very simple questions that do not require additional thought process,
return an empty array containing no elements.

The response must be returned only as an array without any other text.
The first item in the array must be '사용자가 원하는 내용 파악하기'.

---

# Examples

- **Input**: 파이썬으로 웹 크롤러를 만들고 싶은데 어떻게 시작해야 하나요?
- **Output**: ["사용자가 원하는 내용 파악하기", "웹 크롤러에 필요한 라이브러리 탐색하기", "일반적인 웹 크롤러 구현 방법 알아보기", "주의사항 또는 법적 고려사항 확인하기"]

- **Input**: Nest.js에 GraphQL을 적용하고 싶은데 어떻게 해야하나요?
- **Output**: ["사용자가 원하는 내용 파악하기", "GraphQL 라이브러리 탐색하기", "일반적인 GraphQL 적용 방법 알아보기"]

- **Input**: React 상태 관리 라이브러리를 추천해 주세요.
- **Output**: ["사용자가 원하는 내용 파악하기", "React 상태 관리 라이브러리 탐색하기"]

- **Input**: 컴포넌트 분할 전략을 알려주세요.
- **Output**: ["사용자가 원하는 내용 파악하기", "컴포넌트 분할 전략 탐색하기", "클린 아키텍처와 SOLID 원칙 알아보기"]

- **Input**: Next.js API와 Nest.js의 차이점을 설명해 주세요.
- **Output**: ["사용자가 원하는 내용 파악하기", "Next.js API의 사용 목적 알아보기", "Nest.js의 사용 목적 알아보기"]
"""
    
    THINK_STEP_PROMPT = """
# Important

Answer only about '{todo}' in a short and concise manner, always in a single line without any line breaks.

---

# Examples

- **Input**: React 상태 관리 라이브러리를 추천해 주세요.
- **Step**: 사용자가 원하는 내용 파악하기
- **Output**: 사용자가 React 상태 관리 라이브러리를 추천 받고 싶어 하는 것 같습니다. 이를 위해 적절한 라이브러리를 탐색할 필요가 있어보입니다.

- **Input**: Next.js API와 Nest.js의 차이점을 설명해 주세요.
- **Step**: Next.js API의 사용 목적 알아보기
- **Output**: Next.js는 프론트엔드 프레임워크인 React.js를 기반으로 하며, API 라우트를 통해 백엔드 로직을 구현할 수 있습니다. 사용자에게 Nest.js의 사용 목적을 비교하고 설명할 필요가 있어보입니다.

- **Input**: 컴포넌트 분할 전략을 알려주세요.
- **Step**: 클린 아키텍처와 SOLID 원칙 알아보기
- **Output**: 클린 아키텍처는 관심사의 분리(Separation of Concerns)를 통해 유지보수성과 확장성을 높이는 설계 패턴입니다. SOLID 원칙은 객체지향 설계의 다섯 가지 기본 원칙으로, 유지보수성과 유연성을 높이는 데 도움이 됩니다.
"""
    
    FINAL_ANSWER_PROMPT = """
# Important

Write a final answer to the user input based on the thought process below.

---

# Thought process

{thinking_process}
"""

    TOOL_SELECTION_PROMPT = """
# Important

1. get_datetime(location: str) - Get the current datetime in a given location
   - location: The city and state, e.g. "Seoul, Republic of Korea"

2. get_stock_price(symbol: str) - Get the current stock price for a given symbol
   - symbol: The stock symbol, e.g. "AAPL", "MSFT"

When the user asks a question that requires a tool, respond ONLY with a JSON object in the following format:
{"tool": "tool_name", "args": {"arg_name": "arg_value"}}

If no tool is needed, respond with:
{"tool": "none", "response": "your direct answer"}

---

# Examples

- **Input**: "What time is it in Seoul?"
- **Output**: {"tool": "get_datetime", "args": {"location": "Seoul, Republic of Korea"}}

- **Input**: "What's the price of AAPL?"
- **Output**: {"tool": "get_stock_price", "args": {"symbol": "AAPL"}}

- **Input**: "Hello!"
- **Output**: {"tool": "none", "response": "Hello! How can I help you today?"}
"""


# ============================================================================
# Output Event
# ============================================================================

class OutputEvent:
    """출력 이벤트를 나타내는 클래스"""
    
    def __init__(self, mode: OutputMode, event_type: str, content: str = "", is_start: bool = True):
        self.mode = mode
        self.event_type = event_type  # "tag", "stream", "error"
        self.content = content
        self.is_start = is_start
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (API 응답용)"""
        return {
            "type": self.event_type,
            "mode": self.mode.value,
            "is_start": self.is_start,
            "content": self.content
        }
    
    def to_cli_string(self) -> str:
        """CLI 출력용 문자열로 변환"""
        if self.event_type == "stream":
            return self.content
        elif self.event_type == "tag":
            tag_type = "start" if self.is_start else "end"
            if self.content:
                return f"<|{tag_type}(@{self.mode.value})|>{self.content}"
            return f"<|{tag_type}(@{self.mode.value})|>"
        elif self.event_type == "error":
            return f"[ERROR] {self.content}"
        return ""


# ============================================================================
# Thinking Processor
# ============================================================================

class ThinkingProcessor:
    """사고 과정 처리 담당 클래스"""
    
    def __init__(self, llm: ChatLlamaCpp, chat_history: List[BaseMessage]):
        self.llm = llm
        self.chat_history = chat_history
    
    async def generate_todolist_stream(self, user_message: str) -> AsyncGenerator[tuple[OutputEvent, List[str]], None]:
        """투두 리스트 생성 (스트리밍)"""
        messages = [
            SystemMessage(content=f"{PromptBuilder.DEFAULT_SYSTEM_PROMPT}\n\n{PromptBuilder.TODOLIST_PROMPT}"),
            *self.chat_history,
            HumanMessage(content=user_message)
        ]

        string_list_gbnf = r"""
root   ::= "[" ws (string ("," ws string){0,3})? "]" ws
string ::= "\"" (
    [^"\\] | 
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
)* "\"" ws
ws     ::= [ \t\n]
"""
        list_grammar = LlamaGrammar.from_string(string_list_gbnf)
        chain = self.llm.bind(grammar=list_grammar)
        
        # TODOLIST 시작 태그
        yield (OutputEvent(OutputMode.TODOLIST, "tag", is_start=True), [])
        
        full_response = ""
        
        # 스트리밍으로 투두리스트 생성
        async for chunk in chain.astream(messages):
            text = chunk.content
            yield (OutputEvent(OutputMode.TODOLIST, "stream", text), [])
            full_response += text
        
        # TODOLIST 종료 태그
        yield (OutputEvent(OutputMode.TODOLIST, "tag", is_start=False), [])
        
        # JSON 파싱
        try:
            todolist = json.loads(full_response)

            clean_todolist = []
            for todo in todolist:
                if todo.strip() != "":
                    clean_todolist.append(todo.strip())
            todolist = clean_todolist

            if len(clean_todolist) == 0:
                todolist = ["사용자가 원하는 내용 파악하기"]
        except json.JSONDecodeError:
            todolist = ["사용자가 원하는 내용 파악하기"]
        
        # 최종적으로 파싱된 투두리스트 반환
        yield (None, todolist)
    
    async def process_todo_item_stream(self, todo: str, user_message: str) -> AsyncGenerator[tuple[OutputEvent, str], None]:
        """개별 투두 항목 처리 (스트리밍)"""
        system_prompt = PromptBuilder.THINK_STEP_PROMPT.format(todo=todo)
        messages = [
            SystemMessage(content=f"{PromptBuilder.DEFAULT_SYSTEM_PROMPT}\n\n{system_prompt}"),
            *self.chat_history,
            HumanMessage(content=user_message)
        ]
        
        # TODO 시작 태그 (투두 내용 포함)
        yield (OutputEvent(OutputMode.TODO, "tag", todo, is_start=True), "")
        yield (OutputEvent(OutputMode.TODO, "tag", is_start=False), "")
        
        # RESULT 시작 태그
        yield (OutputEvent(OutputMode.RESULT, "tag", is_start=True), "")
        
        full_response = ""
        
        # 스트리밍으로 결과 생성
        async for chunk in self.llm.astream(messages, max_tokens=512):
            text = chunk.content
            yield (OutputEvent(OutputMode.RESULT, "stream", text), "")
            full_response += text
        
        # RESULT 종료 태그
        yield (OutputEvent(OutputMode.RESULT, "tag", is_start=False), "")
        
        # 최종 결과 반환
        yield (None, full_response.strip())


# ============================================================================
# Versatile Agent
# ============================================================================

class VersatileAgent:
    """메인 에이전트 클래스 - CLI와 서버에서 공통으로 사용"""
    
    def __init__(self, model_path: str, llm_model_config: ModelConfig = None, 
                 generation_config: GenerationConfig = None):
        config = llm_model_config or ModelConfig()
        gen_config = generation_config or GenerationConfig()
        
        self.llm = ChatLlamaCpp(
            model_path=model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            n_batch=config.n_batch,
            n_threads=config.n_threads or multiprocessing.cpu_count() - 1,
            use_mlock=config.use_mlock,
            use_mmap=config.use_mmap,
            verbose=config.verbose,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            repeat_penalty=gen_config.repeat_penalty,
            max_tokens=gen_config.max_tokens,
            chat_format="llama-3"
        )
        
        self.sessions: Dict[str, List[BaseMessage]] = {}
    
    def get_history(self, session_id: str = "default") -> List[BaseMessage]:
        """세션별 대화 기록 조회"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]
    
    def clear_history(self, session_id: str = "default"):
        """세션별 대화 기록 초기화"""
        if session_id in self.sessions:
            self.sessions[session_id].clear()
    
    async def chat_stream(self, user_message: str, session_id: str = "default") -> AsyncGenerator[OutputEvent, None]:
        """일반 대화 (스트리밍)"""
        chat_history = self.get_history(session_id)
        
        messages = [
            SystemMessage(content=PromptBuilder.DEFAULT_SYSTEM_PROMPT),
            *chat_history,
            HumanMessage(content=user_message)
        ]
        
        yield OutputEvent(OutputMode.BASIC, "tag", is_start=True)
        
        full_response = ""
        
        async for chunk in self.llm.astream(messages):
            text = chunk.content
            yield OutputEvent(OutputMode.BASIC, "stream", text)
            full_response += text
        
        yield OutputEvent(OutputMode.BASIC, "tag", is_start=False)
        
        chat_history.append(HumanMessage(content=user_message))
        chat_history.append(AIMessage(content=full_response.strip()))
    
    async def tool_stream(self, user_message: str, session_id: str = "default") -> AsyncGenerator[OutputEvent, None]:
        """도구 사용 모드 (스트리밍)"""
        chat_history = self.get_history(session_id)
        
        messages = [
            SystemMessage(content=f"{PromptBuilder.DEFAULT_SYSTEM_PROMPT}\n\n{PromptBuilder.TOOL_SELECTION_PROMPT}"),
            *chat_history,
            HumanMessage(content=user_message)
        ]
        
        # GBNF 문법: JSON 형태의 툴 호출 또는 일반 응답
        tool_call_gbnf = r"""
root ::= "{" ws "\"tool\"" ws ":" ws tool ws "," ws args ws "}"
tool ::= "\"get_datetime\"" | "\"get_stock_price\"" | "\"none\""
args ::= datetime-args | stock-args | response-args

datetime-args ::= "\"args\"" ws ":" ws "{" ws "\"location\"" ws ":" ws string ws "}"
stock-args ::= "\"args\"" ws ":" ws "{" ws "\"symbol\"" ws ":" ws string ws "}"
response-args ::= "\"response\"" ws ":" ws string

string ::= "\"" (
    [^"\\] | 
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
)* "\""
ws ::= [ \t\n]*
"""
        
        tool_grammar = LlamaGrammar.from_string(tool_call_gbnf)
        chain = self.llm.bind(grammar=tool_grammar)
        
        # 툴 선택 및 인자 생성
        full_response = ""
        async for chunk in chain.astream(messages):
            text = chunk.content
            full_response += text
        
        # JSON 파싱
        try:
            tool_call = json.loads(full_response)
            tool_name = tool_call.get("tool")
            
            if tool_name == "none":
                # 툴이 필요없는 경우 바로 응답
                direct_response = tool_call.get("response", "")
                
                yield OutputEvent(OutputMode.BASIC, "tag", is_start=True)
                yield OutputEvent(OutputMode.BASIC, "stream", direct_response)
                yield OutputEvent(OutputMode.BASIC, "tag", is_start=False)
                
                chat_history.append(HumanMessage(content=user_message))
                chat_history.append(AIMessage(content=direct_response))
                return
            
            # 툴 호출
            tool_args = tool_call.get("args", {})
            
            yield OutputEvent(OutputMode.TOOL_CALL, "tag", f"{tool_name}: {json.dumps(tool_args)}", is_start=True)
            yield OutputEvent(OutputMode.TOOL_CALL, "tag", is_start=False)
            
            # 툴 실행
            if tool_name == "get_datetime":
                result = get_datetime.invoke(tool_args)
            elif tool_name == "get_stock_price":
                result = get_stock_price.invoke(tool_args)
            else:
                result = "Unknown tool"
            
            yield OutputEvent(OutputMode.TOOL_RESULT, "tag", result, is_start=True)
            yield OutputEvent(OutputMode.TOOL_RESULT, "tag", is_start=False)
            
            # 툴 결과를 바탕으로 최종 응답 생성
            final_messages = [
                SystemMessage(content=PromptBuilder.DEFAULT_SYSTEM_PROMPT),
                *chat_history,
                HumanMessage(content=user_message),
                AIMessage(content=f"Tool result: {result}"),
                HumanMessage(content=f"Tool result를 참고해서 최종 답변을 제공해 주세요."),
            ]
            
            yield OutputEvent(OutputMode.BASIC, "tag", is_start=True)
            
            final_response = ""
            async for chunk in self.llm.astream(final_messages):
                text = chunk.content
                yield OutputEvent(OutputMode.BASIC, "stream", text)
                final_response += text
            
            yield OutputEvent(OutputMode.BASIC, "tag", is_start=False)
            
            chat_history.append(HumanMessage(content=user_message))
            chat_history.append(AIMessage(content=final_response.strip()))
            
        except json.JSONDecodeError:
            # JSON 파싱 실패시 에러 처리
            yield OutputEvent(OutputMode.BASIC, "tag", is_start=True)
            yield OutputEvent(OutputMode.BASIC, "stream", "Sorry, I encountered an error processing your request.")
            yield OutputEvent(OutputMode.BASIC, "tag", is_start=False)
    
    async def think_and_answer_stream(self, user_message: str, session_id: str = "default") -> AsyncGenerator[OutputEvent, None]:
        """사고 과정을 거친 답변 (실시간 스트리밍)"""
        chat_history = self.get_history(session_id)
        
        processor = ThinkingProcessor(self.llm, chat_history)
        
        # 1단계: 투두리스트 생성 (실시간 스트리밍)
        todolist = []
        async for event, parsed_list in processor.generate_todolist_stream(user_message):
            if event is not None:
                yield event
            else:
                todolist = parsed_list
        
        # 2단계: THINK 시작 태그
        yield OutputEvent(OutputMode.THINK, "tag", is_start=True)
        
        # 3단계: 각 투두 항목 처리 (실시간 스트리밍)
        thinking = []
        for todo in todolist:
            todo_result = ""
            async for event, result in processor.process_todo_item_stream(todo.strip(), user_message):
                if event is not None:
                    yield event
                else:
                    todo_result = result
            
            thinking.append({'todo': todo, 'content': todo_result})
        
        # 4단계: THINK 종료 태그
        yield OutputEvent(OutputMode.THINK, "tag", is_start=False)
        
        # 5단계: 최종 답변 생성
        thinking_text = "\n\n".join(
            f"{item['todo']}: {item['content']}" for item in thinking
        )
        
        system_prompt = PromptBuilder.FINAL_ANSWER_PROMPT.format(
            thinking_process=thinking_text
        )
        
        messages = [
            SystemMessage(content=f"{PromptBuilder.DEFAULT_SYSTEM_PROMPT}\n\n{system_prompt}"),
            *chat_history,
            HumanMessage(content=user_message)
        ]
        
        yield OutputEvent(OutputMode.BASIC, "tag", is_start=True)
        
        full_response = ""
        
        async for chunk in self.llm.astream(messages):
            text = chunk.content
            yield OutputEvent(OutputMode.BASIC, "stream", text)
            full_response += text
        
        yield OutputEvent(OutputMode.BASIC, "tag", is_start=False)
        
        chat_history.append(HumanMessage(content=user_message))
        chat_history.append(AIMessage(content=full_response.strip()))
    
    def get_formatted_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """포맷된 대화 기록 반환"""
        history = self.get_history(session_id)
        
        formatted_history = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                formatted_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_history.append({"role": "assistant", "content": msg.content})
        
        return formatted_history
