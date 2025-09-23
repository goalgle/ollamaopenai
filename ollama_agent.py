import requests
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class Message:
    """메시지 구조체"""
    role: str  # "user", "assistant", "system"
    content: str


class OllamaAgent:
    """
    Ollama 기반 에이전트 클래스
    """

    def __init__(
            self,
            model: str = "llama3.2",
            system_prompt: str = "당신은 도움이 되는 AI 어시스턴트입니다.",
            base_url: str = "http://localhost:11434"
    ):
        """
        OllamaAgent 초기화

        Args:
            model (str): 사용할 Ollama 모델명
            system_prompt (str): 기본 지침 (시스템 프롬프트)
            base_url (str): Ollama 서버 URL
        """
        self.model = model
        self.system_prompt = system_prompt
        self.base_url = base_url
        self.conversation_history: List[Message] = []

        # 시스템 프롬프트를 대화 히스토리에 추가
        if system_prompt:
            self.conversation_history.append(
                Message(role="system", content=system_prompt)
            )

    def _prepare_messages(self, user_input: str) -> List[Dict[str, str]]:
        """
        API 호출을 위한 메시지 형식 준비

        Args:
            user_input (str): 사용자 입력

        Returns:
            List[Dict[str, str]]: API 형식의 메시지 목록
        """
        # 새로운 사용자 메시지 추가
        user_message = Message(role="user", content=user_input)

        # 임시로 메시지 목록 구성 (실제 히스토리에는 아직 추가 안함)
        temp_messages = self.conversation_history + [user_message]

        # API 형식으로 변환
        api_messages = []
        for msg in temp_messages:
            api_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        return api_messages

    def _call_ollama_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Ollama API 호출

        Args:
            messages: API 형식의 메시지 목록

        Returns:
            str: AI 응답

        Raises:
            Exception: API 호출 실패 시
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result["message"]["content"]

        except requests.exceptions.ConnectionError:
            raise Exception(f"Ollama 서버({self.base_url})에 연결할 수 없습니다.")
        except requests.exceptions.Timeout:
            raise Exception("응답 시간이 초과되었습니다.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 호출 중 오류 발생: {e}")
        except KeyError:
            raise Exception("API 응답 형식이 올바르지 않습니다.")

    def chat(self, user_input: str, save_history: bool = True) -> str:
        """
        사용자 입력을 받아 AI 응답을 반환하는 메인 메소드

        Args:
            user_input (str): 사용자 입력
            save_history (bool): 대화 히스토리 저장 여부

        Returns:
            str: AI 응답
        """
        try:
            # 메시지 준비
            messages = self._prepare_messages(user_input)

            # API 호출
            ai_response = self._call_ollama_api(messages)

            # 히스토리 저장
            if save_history:
                self.conversation_history.append(
                    Message(role="user", content=user_input)
                )
                self.conversation_history.append(
                    Message(role="assistant", content=ai_response)
                )

            return ai_response

        except Exception as e:
            return f"❌ 오류: {str(e)}"

    def get_conversation_history(self) -> List[Message]:
        """대화 히스토리 반환"""
        return self.conversation_history.copy()

    def clear_history(self, keep_system_prompt: bool = True):
        """
        대화 히스토리 초기화

        Args:
            keep_system_prompt (bool): 시스템 프롬프트 유지 여부
        """
        if keep_system_prompt and self.system_prompt:
            self.conversation_history = [
                Message(role="system", content=self.system_prompt)
            ]
        else:
            self.conversation_history = []

    def update_system_prompt(self, new_system_prompt: str):
        """
        시스템 프롬프트 업데이트

        Args:
            new_system_prompt (str): 새로운 시스템 프롬프트
        """
        self.system_prompt = new_system_prompt

        # 기존 히스토리에서 시스템 메시지 제거
        self.conversation_history = [
            msg for msg in self.conversation_history
            if msg.role != "system"
        ]

        # 새로운 시스템 프롬프트를 맨 앞에 추가
        if new_system_prompt:
            self.conversation_history.insert(0,
                                             Message(role="system", content=new_system_prompt)
                                             )

    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]

        except Exception:
            return []

    def __str__(self) -> str:
        """문자열 표현"""
        return f"OllamaAgent(model='{self.model}', messages={len(self.conversation_history)})"


# 사용 예제
def main():
    print("🤖 Ollama Agent 테스트")
    print("=" * 50)

    # Agent 생성
    agent = OllamaAgent(
        model="llama3.2",
        system_prompt="당신은 도움이 되는 한국어 AI 코드 전문가입니다. md 규격에 맞추어 응답합니다."
    )

    print(f"📋 Agent 정보: {agent}")
    print(f"🔧 사용 가능한 모델들: {agent.get_available_models()}")
    print("-" * 50)

    # 테스트 대화
    test_messages = [
        "안녕하세요!",
        "파이썬에서 리스트를 역순으로 정렬하는 방법을 알려주세요",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"👤 사용자 ({i}): {message}")
        response = agent.chat(message)
        print(f"🤖 Agent ({i}): {response}")
        print()

    # 히스토리 확인
    print("📚 대화 히스토리:")
    for i, msg in enumerate(agent.get_conversation_history()):
        print(f"  {i+1}. [{msg.role}] {msg.content[:50]}...")

    print(f"\n총 {len(agent.get_conversation_history())} 개의 메시지")


if __name__ == "__main__":
    main()