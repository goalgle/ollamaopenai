import os
import asyncio
from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled

def setup_ollama_environment():
    """
    Ollama를 OpenAI Agents SDK와 연동하기 위한 환경 설정
    """
    # Ollama를 OpenAI API처럼 사용하기 위한 환경 변수 설정
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"  # 더미 키 (Ollama는 API 키가 필요 없음)
    
    # Chat Completions API로 설정 (Ollama 호환)
    set_default_openai_api('chat_completions')
    
    # 트레이싱 비활성화 (OpenAI API 키 관련 에러 방지)
    set_tracing_disabled(True)
    
    print("✅ Ollama 환경 설정 완료")
    print(f"   - Base URL: {os.environ['OPENAI_BASE_URL']}")
    print(f"   - API: Chat Completions (Ollama 호환)")
    print(f"   - Tracing: Disabled")


def create_math_tutor():
    """수학 튜터 에이전트 생성"""
    
    # 환경 설정
    setup_ollama_environment()
    
    # OpenAI Agents SDK를 사용하여 에이전트 생성
    agent = Agent(
        name="Math Tutor",
        instructions="""You are a helpful math tutor. 
        - Provide clear, step-by-step explanations for math problems
        - Use examples to illustrate concepts
        - Be patient and encouraging
        - Answer in Korean when the question is in Korean
        - Show your work and reasoning at each step""",
        model="llama3.2"  # Ollama 모델명
    )
    
    return agent


def create_coding_assistant():
    """코딩 어시스턴트 에이전트 생성"""
    
    setup_ollama_environment()
    
    agent = Agent(
        name="Coding Assistant",
        instructions="""You are an expert programming assistant.
        - Help with code debugging, optimization, and best practices
        - Provide clear explanations with code examples
        - Support multiple programming languages
        - Give practical, working solutions
        - Answer in Korean when the question is in Korean""",
        model="llama3.2"
    )
    
    return agent


def create_creative_writer():
    """창작 도우미 에이전트 생성"""
    
    setup_ollama_environment()
    
    agent = Agent(
        name="Creative Writer",
        instructions="""You are a creative writing assistant.
        - Help with storytelling, character development, and plot ideas
        - Provide inspiration and creative suggestions
        - Adapt writing style to match the user's needs
        - Be imaginative and engaging
        - Write in Korean when requested""",
        model="llama3.2"
    )
    
    return agent


def test_agents():
    """다양한 에이전트들을 테스트"""
    
    print("🤖 OpenAI Agents SDK + Ollama 테스트")
    print("=" * 60)
    
    # 1. 수학 튜터 테스트
    print("\n📐 Math Tutor 테스트")
    print("-" * 30)
    
    try:
        math_agent = create_math_tutor()
        math_question = "2차 방정식 x² - 5x + 6 = 0을 인수분해로 풀어주세요."
        
        print(f"👤 학생: {math_question}")
        result = Runner.run_sync(math_agent, math_question)
        response = result.final_output
        print(f"🧑‍🏫 {math_agent.name}: {response}")
        
    except Exception as e:
        print(f"❌ Math Tutor 오류: {e}")
    
    # 2. 코딩 어시스턴트 테스트  
    print("\n💻 Coding Assistant 테스트")
    print("-" * 30)
    
    try:
        coding_agent = create_coding_assistant()
        coding_question = "Python에서 리스트 컴프리헨션을 사용해서 1부터 10까지 짝수만 제곱하는 코드를 작성해주세요."
        
        print(f"👤 개발자: {coding_question}")
        result = Runner.run_sync(coding_agent, coding_question)
        response = result.final_output
        print(f"👨‍💻 {coding_agent.name}: {response}")
        
    except Exception as e:
        print(f"❌ Coding Assistant 오류: {e}")
    
    # 3. 창작 도우미 테스트
    print("\n✍️ Creative Writer 테스트")
    print("-" * 30)
    
    try:
        writer_agent = create_creative_writer()
        writing_question = "SF 소설의 흥미로운 설정 아이디어를 3가지 제안해주세요."
        
        print(f"👤 작가: {writing_question}")
        result = Runner.run_sync(writer_agent, writing_question)
        response = result.final_output
        print(f"✍️ {writer_agent.name}: {response}")
        
    except Exception as e:
        print(f"❌ Creative Writer 오류: {e}")


def interactive_chat():
    """대화형 채팅 모드"""
    
    print("\n💬 대화형 모드 시작")
    print("사용 가능한 에이전트: math, coding, creative")
    print("종료하려면 'quit' 입력")
    print("-" * 40)
    
    agents = {
        'math': create_math_tutor(),
        'coding': create_coding_assistant(), 
        'creative': create_creative_writer()
    }
    
    current_agent_name = 'math'
    current_agent = agents[current_agent_name]
    
    print(f"현재 에이전트: {current_agent.name}")
    
    while True:
        try:
            user_input = input(f"\n👤 [{current_agent_name}] 당신: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("👋 대화를 종료합니다.")
                break
            
            # 에이전트 변경 명령
            if user_input.startswith('/'):
                command = user_input[1:].lower()
                if command in agents:
                    current_agent_name = command
                    current_agent = agents[current_agent_name]
                    print(f"✅ {current_agent.name}로 변경되었습니다.")
                    continue
                elif command == 'help':
                    print("사용법:")
                    print("  /math - 수학 튜터로 변경")
                    print("  /coding - 코딩 어시스턴트로 변경")
                    print("  /creative - 창작 도우미로 변경")
                    print("  /help - 도움말 보기")
                    continue
                else:
                    print("❌ 알 수 없는 명령어입니다. /help를 입력하세요.")
                    continue
            
            if not user_input:
                continue
            
            # 에이전트 실행
            print(f"🤖 {current_agent.name}: ", end="", flush=True)
            result = Runner.run_sync(current_agent, user_input)
            response = result.final_output
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 대화를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


def main():
    """메인 함수"""
    
    print("🚀 OpenAI Agents SDK + Ollama 시작")
    print("=" * 60)
    
    # 연결 테스트
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Ollama 서버 연결 성공")
            print(f"📋 사용 가능한 모델: {[m['name'] for m in models.get('models', [])]}")
        else:
            print("⚠️ Ollama 서버 응답이 이상합니다.")
    except Exception as e:
        print(f"❌ Ollama 서버 연결 실패: {e}")
        print("해결방법:")
        print("1. 터미널에서 'ollama serve' 실행")
        print("2. 모델 다운로드: 'ollama pull llama3.2'")
        return
    
    # 선택 메뉴
    while True:
        print("\n" + "=" * 60)
        print("선택하세요:")
        print("1. 에이전트 테스트 실행")
        print("2. 대화형 모드")
        print("3. 종료")
        
        try:
            choice = input("선택 (1-3): ").strip()
            
            if choice == "1":
                test_agents()
            elif choice == "2":
                interactive_chat()
            elif choice == "3":
                print("👋 프로그램을 종료합니다.")
                break
            else:
                print("❌ 올바른 번호를 입력하세요.")
                
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break


if __name__ == "__main__":
    main()
