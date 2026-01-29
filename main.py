from typing import List, Dict, Iterator
from dataclasses import dataclass
from enum import Enum
import json

from llama_cpp import Llama

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


class OutputMode(Enum):
    """출력 모드 정의"""
    BASIC = "basic"
    TODOLIST = "todolist"
    THINK = "think"
    TODO = "todo"
    RESULT = "result"
    LOG = "log"


@dataclass
class ModelConfig:
    """LLM 모델 설정"""
    n_ctx: int = 4096
    n_gpu_layers: int = 20
    n_batch: int = 512
    n_threads: int = 6
    n_threads_batch: int = 6
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
    stop: List[str] = None
    
    def __post_init__(self):
        if self.stop is None:
            self.stop = ["<|eot_id|>", "<|end_of_text|>"]


class PromptBuilder:
    """프롬프트 생성 담당 클래스"""
    
    DEFAULT_SYSTEM_PROMPT = "You are name is Versatile (버사타일)."
    
    TODOLIST_PROMPT = """
When you receive user input, break it down into sub-todo lists for step-by-step thinking.
The list should be sorted in logical order,
each sub-todo should be independently answerable,
and the content of sub-todos should not be similar to the user input.

For very simple questions that do not require additional thought process,
return an empty array containing no elements.

The response must be returned only as an array without any other text.
The first item in the array must be '사용자가 원하는 내용 파악하기'.

예시)
사용자 입력: 파이썬으로 웹 크롤러를 만들고 싶은데 어떻게 시작해야 하나요?
답변: ["사용자가 원하는 내용 파악하기", "웹 크롤러에 필요한 라이브러리 탐색하기", "일반적인 웹 크롤러 구현 방법 알아보기", "주의사항 또는 법적 고려사항 확인하기"]

사용자 입력: Nest.js에 GraphQL을 적용하고 싶은데 어떻게 해야하나요?
답변: ["사용자가 원하는 내용 파악하기", "GraphQL 라이브러리 탐색하기", "일반적인 GraphQL 적용 방법 알아보기"]

사용자 입력: React 상태 관리 라이브러리를 추천해 주세요.
답변: ["사용자가 원하는 내용 파악하기", "React 상태 관리 라이브러리 탐색하기"]

사용자 입력: 컴포넌트 분할 전략을 알려주세요.
답변: ["사용자가 원하는 내용 파악하기", "컴포넌트 분할 전략 탐색하기", "클린 아키텍처와 SOLID 원칙 알아보기"]

사용자 입력: Next.js API와 Nest.js의 차이점을 설명해 주세요.
답변: ["사용자가 원하는 내용 파악하기", "Next.js API의 사용 목적 알아보기", "Nest.js의 사용 목적 알아보기"]
"""
    
    THINK_STEP_PROMPT = """
Answer only about '{todo}' in a short and concise manner, always in a single line without any line breaks.

예시)
사용자 입력: React 상태 관리 라이브러리를 추천해 주세요.
단계: 사용자가 원하는 내용 파악하기
답변: 사용자가 React 상태 관리 라이브러리를 추천 받고 싶어 하는 것 같습니다. 이를 위해 적절한 라이브러리를 탐색할 필요가 있어보입니다.

사용자 입력: Next.js API와 Nest.js의 차이점을 설명해 주세요.
단계: Next.js API의 사용 목적 알아보기
답변: Next.js는 프론트엔드 프레임워크인 React.js를 기반으로 하며, API 라우트를 통해 백엔드 로직을 구현할 수 있습니다. 사용자에게 Nest.js의 사용 목적을 비교하고 설명할 필요가 있어보입니다.

사용자 입력: 컴포넌트 분할 전략을 알려주세요.
단계: 클린 아키텍처와 SOLID 원칙 알아보기
답변: 클린 아키텍처는 관심사의 분리(Separation of Concerns)를 통해 유지보수성과 확장성을 높이는 설계 패턴입니다. SOLID 원칙은 객체지향 설계의 다섯 가지 기본 원칙으로, 유지보수성과 유연성을 높이는 데 도움이 됩니다.
"""
    
    FINAL_ANSWER_PROMPT = """
Write a final answer to the user input based on the thought process below.

Thought process)
{thinking_process}
"""
    
    @staticmethod
    def build_from_messages(messages: List[BaseMessage], system_prompt: str = "") -> str:
        """LangChain 메시지 리스트로부터 프롬프트 생성"""
        prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{PromptBuilder.DEFAULT_SYSTEM_PROMPT.strip()}\n\n{system_prompt.strip()}<|eot_id|>"
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                continue
            else:
                role = "user"
            
            content = msg.content.strip()
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        
        return prompt
    
    @staticmethod
    def build(user_message: str, messages: List[BaseMessage], system_prompt: str = "") -> str:
        """현재 사용자 메시지를 포함한 전체 프롬프트 생성"""
        prompt = PromptBuilder.build_from_messages(messages, system_prompt)
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt


class OutputFormatter:
    """출력 포매팅 담당 클래스"""
    
    @staticmethod
    def format_tag(mode: OutputMode, content: str = "", is_start: bool = True) -> str:
        """태그 포맷팅"""
        tag_type = "start" if is_start else "end"
        if content:
            return f"<|{tag_type}(@{mode.value})|>{content}"
        return f"<|{tag_type}(@{mode.value})|>"
    
    @staticmethod
    def print_stream(mode: OutputMode, text: str, end: str = '', flush: bool = True):
        """스트리밍 출력"""
        print(text, end=end, flush=flush)
    
    @staticmethod
    def print_tag(mode: OutputMode, content: str = "", is_start: bool = True, newline: bool = True):
        """태그 출력"""
        tag = OutputFormatter.format_tag(mode, content, is_start)
        print(tag, end='\n' if newline else '')


class LlamaCppLLM(LLM):
    """LangChain과 호환되는 llama-cpp-python LLM 래퍼"""
    
    model_path: str
    llm_config: ModelConfig = ModelConfig()
    gen_config: GenerationConfig = GenerationConfig()
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, model_path: str, llm_model_config: ModelConfig = None, 
                 generation_config: GenerationConfig = None, **kwargs):
        config = llm_model_config or ModelConfig()
        gen_cfg = generation_config or GenerationConfig()
        
        super().__init__(
            model_path=model_path,
            llm_config=config,
            gen_config=gen_cfg,
            **kwargs
        )
        
        self._model = Llama(
            model_path=model_path,
            n_ctx=self.llm_config.n_ctx,
            n_gpu_layers=self.llm_config.n_gpu_layers,
            n_batch=self.llm_config.n_batch,
            n_threads=self.llm_config.n_threads,
            n_threads_batch=self.llm_config.n_threads_batch,
            use_mlock=self.llm_config.use_mlock,
            use_mmap=self.llm_config.use_mmap,
            verbose=self.llm_config.verbose,
        )
    
    @property
    def _llm_type(self) -> str:
        return "llama-cpp"
    
    def _call(self, prompt: str, stop: List[str] = None, **kwargs) -> str:
        """비스트리밍 호출"""
        stop_sequences = stop or self.gen_config.stop
        
        output = self._model(
            prompt,
            max_tokens=kwargs.get('max_tokens', self.gen_config.max_tokens),
            temperature=kwargs.get('temperature', self.gen_config.temperature),
            top_p=kwargs.get('top_p', self.gen_config.top_p),
            top_k=kwargs.get('top_k', self.gen_config.top_k),
            repeat_penalty=kwargs.get('repeat_penalty', self.gen_config.repeat_penalty),
            stop=stop_sequences,
            stream=False
        )
        
        return output['choices'][0]['text']
    
    def _stream(self, prompt: str, stop: List[str] = None, **kwargs) -> Iterator[GenerationChunk]:
        """스트리밍 호출"""
        stop_sequences = stop or self.gen_config.stop
        
        output = self._model(
            prompt,
            max_tokens=kwargs.get('max_tokens', self.gen_config.max_tokens),
            temperature=kwargs.get('temperature', self.gen_config.temperature),
            top_p=kwargs.get('top_p', self.gen_config.top_p),
            top_k=kwargs.get('top_k', self.gen_config.top_k),
            repeat_penalty=kwargs.get('repeat_penalty', self.gen_config.repeat_penalty),
            stop=stop_sequences,
            stream=True
        )
        
        for chunk in output:
            text = chunk['choices'][0]['text']
            yield GenerationChunk(text=text)


class ThinkingProcessor:
    """사고 과정 처리 담당 클래스"""
    
    def __init__(self, llm: LlamaCppLLM, chat_history: InMemoryChatMessageHistory):
        self.llm = llm
        self.chat_history = chat_history
    
    def generate_todolist(self, user_message: str) -> List[str]:
        """투두 리스트 생성"""
        messages = self.chat_history.messages
        prompt = PromptBuilder.build(
            user_message, 
            messages,
            PromptBuilder.TODOLIST_PROMPT
        )
        
        OutputFormatter.print_tag(OutputMode.TODOLIST, is_start=True, newline=False)
        
        full_response = ""
        for chunk in self.llm._stream(
            prompt,
            max_tokens=256,
            temperature=0.3,
            top_p=0.8,
            repeat_penalty=1.1
        ):
            text = chunk.text
            OutputFormatter.print_stream(OutputMode.TODOLIST, text)
            full_response += text
        
        OutputFormatter.print_tag(OutputMode.TODOLIST, is_start=False)
        print()
        
        try:
            return json.loads(full_response)
        except json.JSONDecodeError:
            return []
    
    def process_todo_item(self, todo: str, user_message: str) -> str:
        """개별 투두 항목 처리"""
        system_prompt = PromptBuilder.THINK_STEP_PROMPT.format(todo=todo)
        messages = self.chat_history.messages
        prompt = PromptBuilder.build(user_message, messages, system_prompt)
        
        OutputFormatter.print_tag(OutputMode.TODO, todo, is_start=True, newline=False)
        OutputFormatter.print_tag(OutputMode.TODO, is_start=False)
        OutputFormatter.print_tag(OutputMode.RESULT, is_start=True, newline=False)
        
        full_response = ""
        for chunk in self.llm._stream(prompt, max_tokens=512):
            text = chunk.text
            OutputFormatter.print_stream(OutputMode.RESULT, text)
            full_response += text
        
        OutputFormatter.print_tag(OutputMode.RESULT, is_start=False)
        
        return full_response.strip()
    
    def execute_thinking(self, user_message: str) -> List[Dict[str, str]]:
        """전체 사고 과정 실행"""
        todolist = self.generate_todolist(user_message)
        
        thinking = []
        OutputFormatter.print_tag(OutputMode.THINK, is_start=True)
        
        for index, todo in enumerate(todolist):
            if index > 0:
                print()
            
            result = self.process_todo_item(todo.strip(), user_message)
            thinking.append({'todo': todo, 'content': result})
        
        OutputFormatter.print_tag(OutputMode.THINK, is_start=False)
        print()
        
        return thinking


class ChatBot:
    """메인 챗봇 클래스 (LangChain 적용)"""
    
    def __init__(self, model_path: str, llm_model_config: ModelConfig = None):
        self.llm = LlamaCppLLM(model_path, llm_model_config)
        self.chat_history = InMemoryChatMessageHistory()
    
    def chat(self, user_message: str) -> str:
        """일반 대화"""
        messages = self.chat_history.messages
        prompt = PromptBuilder.build(user_message, messages)
        
        OutputFormatter.print_tag(OutputMode.BASIC, is_start=True, newline=False)
        
        full_response = ""
        for chunk in self.llm._stream(prompt):
            text = chunk.text
            OutputFormatter.print_stream(OutputMode.BASIC, text)
            full_response += text
        
        OutputFormatter.print_tag(OutputMode.BASIC, is_start=False)
        
        self.chat_history.add_user_message(user_message)
        self.chat_history.add_ai_message(full_response.strip())
        
        return full_response
    
    def think_and_answer(self, user_message: str) -> str:
        """사고 과정을 거친 답변"""
        processor = ThinkingProcessor(self.llm, self.chat_history)
        thinking = processor.execute_thinking(user_message)
        
        thinking_text = "\n\n".join(
            f"{item['todo']}: {item['content']}" for item in thinking
        )
        
        system_prompt = PromptBuilder.FINAL_ANSWER_PROMPT.format(
            thinking_process=thinking_text
        )
        
        messages = self.chat_history.messages
        prompt = PromptBuilder.build(user_message, messages, system_prompt)
        
        OutputFormatter.print_tag(OutputMode.BASIC, is_start=True, newline=False)
        
        full_response = ""
        for chunk in self.llm._stream(prompt):
            text = chunk.text
            OutputFormatter.print_stream(OutputMode.BASIC, text)
            full_response += text
        
        OutputFormatter.print_tag(OutputMode.BASIC, is_start=False)
        print()
        
        self.chat_history.add_user_message(user_message)
        self.chat_history.add_ai_message(full_response.strip())
        
        return full_response
    
    def clear_history(self):
        """대화 기록 초기화"""
        self.chat_history.clear()
        OutputFormatter.print_tag(OutputMode.LOG, "conversation history cleared.", is_start=True)
        OutputFormatter.print_tag(OutputMode.LOG, is_start=False)
    
    def get_history(self) -> List[BaseMessage]:
        """대화 기록 조회"""
        return self.chat_history.messages


def main():
    """메인 실행 함수"""

    # llama-3.2-Korean-Bllossom-3B
    # model_path = './models/llama-3.2-Korean-Bllossom-3B/f16.gguf'
    # model_path = './models/llama-3.2-Korean-Bllossom-3B/Q8_0.gguf'
    
    # llama-3-Korean-Bllossom-8B
    # model_path = './models/llama-3-Korean-Bllossom-8B/f16.gguf'
    model_path = './models/llama-3-Korean-Bllossom-8B/Q8_0.gguf'
    # model_path = './models/llama-3-Korean-Bllossom-8B/Q4_K_M.gguf'
    # model_path = './models/llama-3-Korean-Bllossom-8B/Open-Ko Q4_K_M.gguf'
    
    # llama-3-Korean-Bllossom-70B
    # model_path = './models/llama-3-Korean-Bllossom-70B/Q4_K_M.gguf'

    # aya-expanse-8b-abliterated
    # model_path = './models/aya-expanse-8b-abliterated/Q8_0.gguf'
    
    chatbot = ChatBot(model_path)
    
    print("=" * 40)
    print("버사타일 (Versatile)")
    print()
    print("commands: 'quit', 'clear'")
    print()
    print("qna: '<prompt>'")
    print("incident mode: '@think <prompt>'")
    print("=" * 40)
    print()
    
    while True:
        user_input = input("you: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            OutputFormatter.print_tag(OutputMode.LOG, "close versatile", is_start=True, newline=False)
            OutputFormatter.print_tag(OutputMode.LOG, is_start=False)
            break
        
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            continue
        
        if user_input.lower().startswith('@think '):
            OutputFormatter.print_tag(OutputMode.LOG, "incident mode enabled.", is_start=True, newline=False)
            OutputFormatter.print_tag(OutputMode.LOG, is_start=False)
            print()
            chatbot.think_and_answer(user_input[7:])
            continue
        
        print()
        chatbot.chat(user_input)
        print()


if __name__ == "__main__":
    main()