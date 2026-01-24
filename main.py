from llama_cpp import Llama
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


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
    n_ctx: int = 16384
    n_gpu_layers: int = -1
    n_batch: int = 8192
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
    
    DEFAULT_SYSTEM_PROMPT = "당신은 버사타일 (Versatile) AI 어시스턴트입니다."
    
    TODOLIST_PROMPT = """
사용자 입력을 받으면, 단계별 사고를 위한 하위 투두 리스트로 분해하세요.
리스트는 논리적 순서대로 정렬되어있어야 하고,
각 하위 투두는 독립적으로 답변이 가능해야 하며,
하위 투두의 내용은 사용자 입력과 비슷하지 않아야 합니다.

추가적인 사고 과정이 필요 없는 매우 간단한 질문의 경우,
아무런 요소도 포함하지 않은 빈 배열을 반환하세요.

답변은 다른 텍스트 없이 무조건 배열로만 반환하세요.
배열 첫번째 순서는 '사용자가 원하는 내용 파악하기' 이어야 합니다.

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
사용자 입력을 받으면, 오직 '{todo}' 단계만 수행하십시오.

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
당신은 단계별 사고를 완료했습니다.
아래 사고 과정을 바탕으로 사용자 질문에 대한 최종 답변을 작성하십시오.

사고 과정:
{thinking_process}
"""
    
    @staticmethod
    def build(user_message: str, conversation_history: List[Dict], 
              system_prompt: str = "") -> str:
        """대화 프롬프트 생성"""
        prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{PromptBuilder.DEFAULT_SYSTEM_PROMPT.strip()}\n\n{system_prompt.strip()}<|eot_id|>"
        
        for msg in conversation_history:
            role = msg['role']
            content = msg['content'].strip()
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        
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


class LLMEngine:
    """LLM 모델 실행 엔진"""
    
    def __init__(self, model_path: str, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = Llama(
            model_path=model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            n_threads_batch=self.config.n_threads_batch,
            use_mlock=self.config.use_mlock,
            use_mmap=self.config.use_mmap,
            verbose=self.config.verbose,
        )
    
    def generate(self, prompt: str, config: GenerationConfig, 
                 stream_callback=None) -> str:
        """텍스트 생성"""
        output = self.model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop,
            stream=True
        )
        
        full_response = ""
        for chunk in output:
            text = chunk['choices'][0]['text']
            if stream_callback:
                stream_callback(text)
            full_response += text
        
        return full_response


class ThinkingProcessor:
    """사고 과정 처리 담당 클래스"""
    
    def __init__(self, engine: LLMEngine, conversation_history: List[Dict]):
        self.engine = engine
        self.conversation_history = conversation_history
    
    def generate_todolist(self, user_message: str) -> List[str]:
        """투두 리스트 생성"""
        prompt = PromptBuilder.build(
            user_message, 
            self.conversation_history,
            PromptBuilder.TODOLIST_PROMPT
        )
        
        config = GenerationConfig(
            max_tokens=256,
            temperature=0.3,
            top_p=0.8,
            repeat_penalty=1.1
        )
        
        OutputFormatter.print_tag(OutputMode.TODOLIST, is_start=True, newline=False)
        
        response = self.engine.generate(
            prompt, 
            config,
            lambda text: OutputFormatter.print_stream(OutputMode.TODOLIST, text)
        )
        
        OutputFormatter.print_tag(OutputMode.TODOLIST, is_start=False)
        print()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []
    
    def process_todo_item(self, todo: str, user_message: str) -> str:
        """개별 투두 항목 처리"""
        system_prompt = PromptBuilder.THINK_STEP_PROMPT.format(
            todo=todo
        )
        
        prompt = PromptBuilder.build(user_message, self.conversation_history, system_prompt)
        
        config = GenerationConfig(max_tokens=512)
        
        OutputFormatter.print_tag(OutputMode.TODO, todo, is_start=True, newline=False)
        OutputFormatter.print_tag(OutputMode.TODO, is_start=False)

        OutputFormatter.print_tag(OutputMode.RESULT, is_start=True, newline=False)
        
        response = self.engine.generate(
            prompt,
            config,
            lambda text: OutputFormatter.print_stream(OutputMode.RESULT, text)
        )
        
        OutputFormatter.print_tag(OutputMode.RESULT, is_start=False)
        
        return response.strip()
    
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
    """메인 챗봇 클래스"""
    
    def __init__(self, model_path: str, model_config: ModelConfig = None):
        self.engine = LLMEngine(model_path, model_config)
        self.conversation_history = []
    
    def chat(self, user_message: str) -> str:
        """일반 대화"""
        prompt = PromptBuilder.build(user_message, self.conversation_history)
        config = GenerationConfig()
        
        OutputFormatter.print_tag(OutputMode.BASIC, is_start=True, newline=False)
        
        response = self.engine.generate(
            prompt,
            config,
            lambda text: OutputFormatter.print_stream(OutputMode.BASIC, text)
        )
        
        OutputFormatter.print_tag(OutputMode.BASIC, is_start=False)
        
        self._update_history(user_message, response.strip())
        return response
    
    def think_and_answer(self, user_message: str) -> str:
        """사고 과정을 거친 답변"""
        processor = ThinkingProcessor(self.engine, self.conversation_history)
        thinking = processor.execute_thinking(user_message)
        
        thinking_text = "\n\n".join(
            f"{item['todo']}: {item['content']}" for item in thinking
        )
        
        system_prompt = PromptBuilder.FINAL_ANSWER_PROMPT.format(
            thinking_process=thinking_text
        )
        
        prompt = PromptBuilder.build(user_message, self.conversation_history, system_prompt)
        config = GenerationConfig()
        
        OutputFormatter.print_tag(OutputMode.BASIC, is_start=True, newline=False)
        
        response = self.engine.generate(
            prompt,
            config,
            lambda text: OutputFormatter.print_stream(OutputMode.BASIC, text)
        )
        
        OutputFormatter.print_tag(OutputMode.BASIC, is_start=False)
        print()
        
        self._update_history(user_message, response.strip())
        return response
    
    def clear_history(self):
        """대화 기록 초기화"""
        self.conversation_history.clear()
        OutputFormatter.print_tag(OutputMode.LOG, "conversation history cleared.", is_start=True)
        OutputFormatter.print_tag(OutputMode.LOG, is_start=False)
    
    def _update_history(self, user_message: str, assistant_response: str):
        """대화 기록 업데이트"""
        self.conversation_history.append({'role': 'user', 'content': user_message})
        self.conversation_history.append({'role': 'assistant', 'content': assistant_response})


def main():
    """메인 실행 함수"""
    model_path = './models/llama-3.2-Korean-Bllossom-3B.Q8_0.gguf'
    # model_path = './models/llama-3.2-Korean-Bllossom-AICA-5B_f16.gguf'
    # model_path = './models/llama-3-Korean-Bllossom-8B.f16.gguf'
    
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
