import requests
import json

def chat_with_ollama(message, model="llama3.2"):
    """
    로컬 Ollama에 메시지를 보내고 응답을 받는 함수

    Args:
        message (str): 보낼 메시지
        model (str): 사용할 모델명 (기본값: llama3.2)

    Returns:
        str: Ollama의 응답
    """
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "stream": False  # 스트리밍 비활성화 (한번에 전체 응답 받기)
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # HTTP 에러 체크

        result = response.json()
        return result["message"]["content"]

    except requests.exceptions.ConnectionError:
        return "❌ Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요."
    except requests.exceptions.RequestException as e:
        return f"❌ 요청 중 오류 발생: {e}"
    except KeyError:
        return "❌ 응답 형식이 예상과 다릅니다."
    except Exception as e:
        return f"❌ 예상치 못한 오류: {e}"

def display_ollama_list():
    # 사용 가능한 모델 확인
    try:
        models_response = requests.get("http://localhost:11434/api/tags")
        if models_response.status_code == 200:
            models = models_response.json()
            print(f"📋 사용 가능한 모델들:")
            for model in models.get("models", []):
                print(f"  - {model['name']}")
            print()
        else:
            print("⚠️  모델 목록을 가져올 수 없습니다.")
    except:
        print("⚠️  Ollama 서버 상태를 확인할 수 없습니다.")

def main():
    print("🤖 로컬 Ollama와 채팅 테스트")
    print("-" * 40)

    display_ollama_list()

    # 메시지 보내기
    message = "안녕"
    print(f"👤 사용자: {message}")

    # Ollama에 메시지 보내고 응답 받기
    response = chat_with_ollama(message)
    print(f"🤖 Ollama: {response}")

if __name__ == "__main__":
    main()