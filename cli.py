import asyncio
from agent import (
    VersatileAgent,
    OutputMode,
    OutputEvent
)


class CLIInterface:
    """CLI 출력 처리 클래스"""
    
    def __init__(self, agent: VersatileAgent):
        self.agent = agent
    
    async def process_stream(self, stream_generator, newline_at_end: bool = True):
        """스트림 이벤트를 CLI에 출력"""
        async for event in stream_generator:
            cli_string = event.to_cli_string()
            
            if event.event_type == "stream":
                print(cli_string, end='', flush=True)
            else:
                print(cli_string, end='', flush=True)
        
        if newline_at_end:
            print()
    
    async def chat(self, user_message: str, session_id: str = "default"):
        """일반 대화"""
        stream = self.agent.chat_stream(user_message, session_id)
        await self.process_stream(stream)
    
    async def tool(self, user_message: str, session_id: str = "default"):
        """도구 사용 모드"""
        stream = self.agent.tool_stream(user_message, session_id)
        await self.process_stream(stream)
    
    async def think(self, user_message: str, session_id: str = "default"):
        """사고 과정 모드"""
        stream = self.agent.think_and_answer_stream(user_message, session_id)
        await self.process_stream(stream)
    
    def clear_history(self, session_id: str = "default"):
        """대화 기록 초기화"""
        self.agent.clear_history(session_id)
        event = OutputEvent(OutputMode.LOG, "tag", "conversation history cleared.", is_start=True)
        print(event.to_cli_string())
        event = OutputEvent(OutputMode.LOG, "tag", is_start=False)
        print(event.to_cli_string())


async def main():
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
    
    print("Loading model...")
    agent = VersatileAgent(model_path)
    cli = CLIInterface(agent)
    print("Model loaded successfully!")
    print()
    
    print("=" * 40)
    print("버사타일 (Versatile)")
    print()
    print("commands: 'quit', 'clear'")
    print()
    print("qna: '<prompt>'")
    print("tool mode: '@tool <prompt>'")
    print("think mode: '@think <prompt>'")
    print("=" * 40)
    print()
    
    session_id = "default"
    
    while True:
        try:
            user_input = input("you: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                event = OutputEvent(OutputMode.LOG, "tag", "close versatile", is_start=True)
                print(event.to_cli_string(), end='')
                event = OutputEvent(OutputMode.LOG, "tag", is_start=False)
                print(event.to_cli_string())
                break
            
            if user_input.lower() == 'clear':
                cli.clear_history(session_id)
                continue
            
            print()
            
            if user_input.lower().startswith('@think '):
                event = OutputEvent(OutputMode.LOG, "tag", "think mode enabled.", is_start=True)
                print(event.to_cli_string(), end='')
                event = OutputEvent(OutputMode.LOG, "tag", is_start=False)
                print(event.to_cli_string())
                print()
                await cli.think(user_input[7:], session_id)
                continue
            
            if user_input.lower().startswith('@tool '):
                event = OutputEvent(OutputMode.LOG, "tag", "tool mode enabled.", is_start=True)
                print(event.to_cli_string(), end='')
                event = OutputEvent(OutputMode.LOG, "tag", is_start=False)
                print(event.to_cli_string())
                print()
                await cli.tool(user_input[6:], session_id)
                print()
                continue
            
            await cli.chat(user_input, session_id)
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
