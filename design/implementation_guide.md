# Implementation Guide: OpenAI SDK + Ollama Integration

## Project Structure

```
ollama-agents/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── environment.py      # Ollama environment setup
│   │   ├── agents.py          # Agent factory and creation
│   │   └── chat.py            # Chat management
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── testing.py         # Testing utilities
│   │   ├── validation.py      # Input validation
│   │   └── performance.py     # Performance monitoring
│   └── config/
│       ├── __init__.py
│       ├── settings.py        # Configuration management
│       └── defaults.py        # Default configurations
├── tests/
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_agents.py
│   ├── test_chat.py
│   └── test_integration.py
├── examples/
│   ├── basic_usage.py
│   ├── interactive_chat.py
│   └── custom_agents.py
├── docs/
│   ├── api_reference.md
│   ├── user_guide.md
│   └── troubleshooting.md
├── requirements.txt
├── setup.py
└── README.md
```

## Implementation Steps

### Phase 1: Core Infrastructure

#### 1.1 Environment Setup Module (`src/core/environment.py`)

```python
import os
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
from agents import set_default_openai_api, set_tracing_disabled

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    default_model: str = "llama3.2"
    timeout: int = 30

class OllamaEnvironment:
    """Manages Ollama-OpenAI compatibility configuration"""

    @staticmethod
    def setup(config: Optional[OllamaConfig] = None) -> bool:
        """Configure OpenAI SDK for Ollama compatibility"""
        if config is None:
            config = OllamaConfig()

        try:
            os.environ["OPENAI_BASE_URL"] = config.base_url
            os.environ["OPENAI_API_KEY"] = config.api_key

            set_default_openai_api('chat_completions')
            set_tracing_disabled(True)

            return True
        except Exception as e:
            raise EnvironmentError(f"Failed to setup Ollama environment: {e}")

    @staticmethod
    def validate_connection(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
        """Validate Ollama server connection"""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get('models', [])
            return {
                'connected': True,
                'models': [m['name'] for m in models],
                'model_count': len(models),
                'server_status': 'healthy'
            }
        except requests.RequestException as e:
            return {
                'connected': False,
                'error': str(e),
                'server_status': 'unreachable'
            }
```

#### 1.2 Agent Factory Module (`src/core/agents.py`)

```python
from agents import Agent
from typing import Optional, List, Dict, Any
from .environment import OllamaEnvironment

class AgentFactory:
    """Factory for creating specialized AI agents"""

    @staticmethod
    def create_math_tutor(
        model: str = "llama3.2",
        language: str = "auto",
        temperature: float = 0.3
    ) -> Agent:
        """Create mathematics tutor agent"""

        instructions = """You are an expert mathematics tutor with the following capabilities:

        CORE RESPONSIBILITIES:
        - Provide clear, step-by-step solutions to mathematical problems
        - Explain mathematical concepts in accessible language
        - Show multiple solution methods when applicable
        - Verify answers and explain the reasoning
        - Generate practice problems for skill development

        TEACHING APPROACH:
        - Break complex problems into manageable steps
        - Use visual representations when helpful
        - Connect new concepts to previously learned material
        - Encourage mathematical reasoning and pattern recognition
        - Adapt explanations to the user's apparent skill level

        LANGUAGE HANDLING:
        - Respond in Korean when the question is in Korean
        - Use appropriate mathematical notation and terminology
        - Provide clear explanations regardless of language

        PROBLEM-SOLVING FORMAT:
        1. Problem analysis and approach identification
        2. Step-by-step solution with clear reasoning
        3. Verification of the answer
        4. Alternative methods (when applicable)
        5. Related concepts or practice suggestions
        """

        OllamaEnvironment.setup()

        return Agent(
            name="Math Tutor",
            instructions=instructions,
            model=model,
            temperature=temperature
        )

    @staticmethod
    def create_coding_assistant(
        model: str = "llama3.2",
        languages: Optional[List[str]] = None,
        temperature: float = 0.5
    ) -> Agent:
        """Create programming assistance agent"""

        supported_languages = languages or [
            "Python", "JavaScript", "Java", "C++", "C#", "Go",
            "Rust", "TypeScript", "SQL", "HTML/CSS"
        ]

        instructions = f"""You are an expert programming assistant specializing in:

        SUPPORTED LANGUAGES: {', '.join(supported_languages)}

        CORE CAPABILITIES:
        - Code debugging and error resolution
        - Performance optimization recommendations
        - Best practices and design patterns
        - Code review and quality assessment
        - Algorithm explanation and implementation
        - API integration guidance
        - Testing strategies and implementation

        RESPONSE FORMAT:
        - Provide working code examples
        - Include clear explanations of logic
        - Suggest improvements and alternatives
        - Highlight potential issues or edge cases
        - Reference relevant documentation when helpful

        CODE QUALITY FOCUS:
        - Readability and maintainability
        - Performance considerations
        - Security best practices
        - Error handling patterns
        - Documentation and comments

        PROBLEM-SOLVING APPROACH:
        1. Understand the requirements thoroughly
        2. Identify the most appropriate solution approach
        3. Provide clean, well-commented code
        4. Explain the reasoning behind design choices
        5. Suggest testing approaches and edge cases
        """

        OllamaEnvironment.setup()

        return Agent(
            name="Coding Assistant",
            instructions=instructions,
            model=model,
            temperature=temperature
        )

    @staticmethod
    def create_creative_writer(
        model: str = "llama3.2",
        writing_style: str = "adaptive",
        temperature: float = 0.8
    ) -> Agent:
        """Create creative writing assistant"""

        instructions = """You are a creative writing assistant with expertise in:

        CREATIVE DOMAINS:
        - Storytelling and narrative development
        - Character creation and development
        - Plot structure and pacing
        - Dialogue writing and voice
        - World-building and setting design
        - Poetry and experimental writing

        WRITING SUPPORT:
        - Generate creative prompts and ideas
        - Develop story outlines and structures
        - Create compelling characters with depth
        - Suggest plot twists and conflict resolution
        - Provide feedback on writing samples
        - Adapt tone and style to match requirements

        STYLE ADAPTATION:
        - Match the user's preferred writing style
        - Adjust complexity and vocabulary appropriately
        - Maintain consistency in tone and voice
        - Respect genre conventions while encouraging creativity

        CREATIVE PROCESS:
        1. Understand the creative vision and goals
        2. Brainstorm ideas and possibilities
        3. Develop concepts with rich detail
        4. Provide structured suggestions and alternatives
        5. Encourage experimentation and personal voice

        GENRES AND FORMATS:
        - Fiction (all genres), Non-fiction, Poetry
        - Screenplays, Short stories, Novels
        - Blog posts, Articles, Marketing copy
        - Game narratives, Interactive fiction
        """

        OllamaEnvironment.setup()

        return Agent(
            name="Creative Writer",
            instructions=instructions,
            model=model,
            temperature=temperature
        )

    @staticmethod
    def create_custom_agent(
        name: str,
        instructions: str,
        model: str = "llama3.2",
        temperature: float = 0.7,
        **kwargs
    ) -> Agent:
        """Create custom agent with specific configuration"""

        OllamaEnvironment.setup()

        return Agent(
            name=name,
            instructions=instructions,
            model=model,
            temperature=temperature,
            **kwargs
        )
```

#### 1.3 Chat Management Module (`src/core/chat.py`)

```python
from agents import Agent, Runner
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConversationResult:
    agent_name: str
    user_input: str
    final_output: str
    execution_time: float
    success: bool
    timestamp: datetime
    error: Optional[str] = None

class ChatManager:
    """Manages interactive conversations with multiple agents"""

    def __init__(self, agents: Dict[str, Agent], default_agent: str = "math"):
        self.agents = agents
        self.current_agent_name = default_agent
        self.conversation_history: List[ConversationResult] = []

        if default_agent not in agents:
            raise ValueError(f"Default agent '{default_agent}' not found in available agents")

    @property
    def current_agent(self) -> Agent:
        return self.agents[self.current_agent_name]

    def switch_agent(self, agent_name: str) -> bool:
        """Switch to different agent"""
        if agent_name in self.agents:
            self.current_agent_name = agent_name
            return True
        return False

    def process_message(
        self,
        message: str,
        agent_name: Optional[str] = None
    ) -> ConversationResult:
        """Process message with specified or current agent"""

        target_agent_name = agent_name or self.current_agent_name
        target_agent = self.agents[target_agent_name]

        start_time = datetime.now()

        try:
            result = Runner.run_sync(target_agent, message)
            end_time = datetime.now()

            execution_time = (end_time - start_time).total_seconds()

            conversation_result = ConversationResult(
                agent_name=target_agent_name,
                user_input=message,
                final_output=result.final_output,
                execution_time=execution_time,
                success=True,
                timestamp=start_time
            )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            conversation_result = ConversationResult(
                agent_name=target_agent_name,
                user_input=message,
                final_output="",
                execution_time=execution_time,
                success=False,
                timestamp=start_time,
                error=str(e)
            )

        self.conversation_history.append(conversation_result)
        return conversation_result

    def start_interactive_session(self) -> None:
        """Start interactive chat session"""

        print("🤖 Interactive AI Chat Session")
        print("=" * 50)
        print(f"Available agents: {', '.join(self.agents.keys())}")
        print(f"Current agent: {self.current_agent_name}")
        print("\nCommands:")
        print("  /switch <agent> - Switch to different agent")
        print("  /history - Show conversation history")
        print("  /clear - Clear conversation history")
        print("  /help - Show this help message")
        print("  /quit - Exit session")
        print("-" * 50)

        while True:
            try:
                user_input = input(f"\n[{self.current_agent_name}] You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if self._handle_command(user_input):
                        continue
                    else:
                        break

                # Process regular message
                print(f"🤖 {self.current_agent.name}: ", end="", flush=True)
                result = self.process_message(user_input)

                if result.success:
                    print(result.final_output)
                else:
                    print(f"❌ Error: {result.error}")

            except KeyboardInterrupt:
                print("\n👋 Session ended by user")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")

    def _handle_command(self, command: str) -> bool:
        """Handle chat commands. Returns True to continue, False to exit"""

        parts = command[1:].split()
        cmd = parts[0].lower()

        if cmd == "quit" or cmd == "exit":
            print("👋 Goodbye!")
            return False

        elif cmd == "switch" and len(parts) > 1:
            agent_name = parts[1]
            if self.switch_agent(agent_name):
                print(f"✅ Switched to {self.agents[agent_name].name}")
            else:
                print(f"❌ Agent '{agent_name}' not found")

        elif cmd == "history":
            self._show_history()

        elif cmd == "clear":
            self.conversation_history.clear()
            print("✅ Conversation history cleared")

        elif cmd == "help":
            self._show_help()

        else:
            print(f"❌ Unknown command: {command}")
            print("Type /help for available commands")

        return True

    def _show_history(self) -> None:
        """Display conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history")
            return

        print(f"\n📝 Conversation History ({len(self.conversation_history)} messages)")
        print("-" * 50)

        for i, result in enumerate(self.conversation_history[-10:], 1):
            status = "✅" if result.success else "❌"
            print(f"{i}. [{result.agent_name}] {status} {result.user_input[:50]}...")

    def _show_help(self) -> None:
        """Show help information"""
        print("\n🆘 Available Commands:")
        print("  /switch <agent> - Switch to different agent")
        print("  /history - Show recent conversation history")
        print("  /clear - Clear conversation history")
        print("  /help - Show this help message")
        print("  /quit - Exit session")
        print(f"\nAvailable agents: {', '.join(self.agents.keys())}")
```

### Phase 2: Testing Infrastructure

#### 2.1 Testing Utilities (`src/utils/testing.py`)

```python
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import time
from agents import Agent, Runner

@dataclass
class TestResult:
    test_name: str
    agent_name: str
    input_query: str
    success: bool
    response_time: float
    response_length: int
    timestamp: datetime
    error: Optional[str] = None

@dataclass
class TestSuiteResult:
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    avg_response_time: float
    detailed_results: List[TestResult]

class AgentTester:
    """Comprehensive testing suite for AI agents"""

    DEFAULT_TEST_SCENARIOS = {
        'math': [
            "Solve the equation x² - 5x + 6 = 0",
            "What is the derivative of x³ + 2x² - 5x + 1?",
            "Calculate the area of a circle with radius 7",
            "Explain the Pythagorean theorem with an example"
        ],
        'coding': [
            "Write a Python function to check if a number is prime",
            "How do I implement a binary search algorithm?",
            "Debug this code: for i in range(10) print(i)",
            "Explain the difference between lists and tuples in Python"
        ],
        'creative': [
            "Write a short story opening about a mysterious door",
            "Create a character profile for a sci-fi protagonist",
            "Suggest plot ideas for a mystery novel",
            "Write a haiku about autumn"
        ]
    }

    @classmethod
    def run_comprehensive_tests(
        cls,
        agents: Dict[str, Agent],
        custom_scenarios: Optional[Dict[str, List[str]]] = None
    ) -> TestSuiteResult:
        """Run comprehensive test suite on all agents"""

        test_scenarios = custom_scenarios or cls.DEFAULT_TEST_SCENARIOS
        all_results = []

        for agent_name, agent in agents.items():
            if agent_name in test_scenarios:
                scenarios = test_scenarios[agent_name]
                results = cls._test_agent(agent, agent_name, scenarios)
                all_results.extend(results)

        return cls._compile_results(all_results)

    @classmethod
    def _test_agent(
        cls,
        agent: Agent,
        agent_name: str,
        test_queries: List[str]
    ) -> List[TestResult]:
        """Test individual agent with provided queries"""

        results = []

        for query in test_queries:
            start_time = time.time()

            try:
                result = Runner.run_sync(agent, query)
                end_time = time.time()

                test_result = TestResult(
                    test_name=f"{agent_name}_query_{len(results)+1}",
                    agent_name=agent_name,
                    input_query=query,
                    success=True,
                    response_time=end_time - start_time,
                    response_length=len(result.final_output),
                    timestamp=datetime.now()
                )

            except Exception as e:
                end_time = time.time()

                test_result = TestResult(
                    test_name=f"{agent_name}_query_{len(results)+1}",
                    agent_name=agent_name,
                    input_query=query,
                    success=False,
                    response_time=end_time - start_time,
                    response_length=0,
                    timestamp=datetime.now(),
                    error=str(e)
                )

            results.append(test_result)

        return results

    @classmethod
    def _compile_results(cls, results: List[TestResult]) -> TestSuiteResult:
        """Compile individual test results into suite summary"""

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        successful_results = [r for r in results if r.success]
        avg_response_time = (
            sum(r.response_time for r in successful_results) / len(successful_results)
            if successful_results else 0
        )

        return TestSuiteResult(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            detailed_results=results
        )
```

### Phase 3: Example Implementation

#### 3.1 Basic Usage Example (`examples/basic_usage.py`)

```python
#!/usr/bin/env python3
"""
Basic usage example for OpenAI SDK + Ollama integration
"""

from src.core.environment import OllamaEnvironment
from src.core.agents import AgentFactory
from src.utils.testing import AgentTester
from agents import Runner

def main():
    print("🚀 OpenAI SDK + Ollama Basic Usage Example")
    print("=" * 60)

    # 1. Setup environment
    print("\n1. Setting up Ollama environment...")
    try:
        OllamaEnvironment.setup()
        connection_info = OllamaEnvironment.validate_connection()

        if connection_info['connected']:
            print(f"✅ Connected to Ollama server")
            print(f"📋 Available models: {connection_info['models']}")
        else:
            print(f"❌ Connection failed: {connection_info['error']}")
            return

    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return

    # 2. Create agents
    print("\n2. Creating specialized agents...")

    try:
        math_agent = AgentFactory.create_math_tutor()
        coding_agent = AgentFactory.create_coding_assistant()
        writer_agent = AgentFactory.create_creative_writer()

        print("✅ Math Tutor created")
        print("✅ Coding Assistant created")
        print("✅ Creative Writer created")

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return

    # 3. Test basic functionality
    print("\n3. Testing basic functionality...")

    test_cases = [
        (math_agent, "Solve x² - 4x + 3 = 0", "Math Tutor"),
        (coding_agent, "Write a Python function to reverse a string", "Coding Assistant"),
        (writer_agent, "Write a creative opening for a mystery story", "Creative Writer")
    ]

    for agent, query, name in test_cases:
        print(f"\n📤 Testing {name}")
        print(f"Query: {query}")

        try:
            result = Runner.run_sync(agent, query)
            print(f"✅ Response received ({len(result.final_output)} characters)")
            print(f"📝 Preview: {result.final_output[:100]}...")

        except Exception as e:
            print(f"❌ Error: {e}")

    # 4. Run comprehensive tests
    print("\n4. Running comprehensive test suite...")

    agents = {
        'math': math_agent,
        'coding': coding_agent,
        'creative': writer_agent
    }

    try:
        test_results = AgentTester.run_comprehensive_tests(agents)

        print(f"📊 Test Results:")
        print(f"   Total tests: {test_results.total_tests}")
        print(f"   Passed: {test_results.passed_tests}")
        print(f"   Failed: {test_results.failed_tests}")
        print(f"   Success rate: {test_results.success_rate:.1f}%")
        print(f"   Avg response time: {test_results.avg_response_time:.2f}s")

    except Exception as e:
        print(f"❌ Test suite failed: {e}")

    print("\n🎉 Basic usage example completed!")

if __name__ == "__main__":
    main()
```

#### 3.2 Interactive Chat Example (`examples/interactive_chat.py`)

```python
#!/usr/bin/env python3
"""
Interactive chat example with multiple AI agents
"""

from src.core.environment import OllamaEnvironment
from src.core.agents import AgentFactory
from src.core.chat import ChatManager

def main():
    print("💬 Interactive AI Chat Example")
    print("=" * 50)

    # Setup environment
    try:
        OllamaEnvironment.setup()
        connection_info = OllamaEnvironment.validate_connection()

        if not connection_info['connected']:
            print(f"❌ Cannot connect to Ollama: {connection_info['error']}")
            print("\nPlease ensure:")
            print("1. Ollama is installed and running (ollama serve)")
            print("2. Required models are downloaded (ollama pull llama3.2)")
            return

    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return

    # Create agents
    try:
        agents = {
            'math': AgentFactory.create_math_tutor(),
            'coding': AgentFactory.create_coding_assistant(),
            'creative': AgentFactory.create_creative_writer()
        }

        print("✅ All agents created successfully")

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return

    # Start interactive session
    chat_manager = ChatManager(agents, default_agent='math')
    chat_manager.start_interactive_session()

if __name__ == "__main__":
    main()
```

### Phase 4: Configuration and Testing Files

#### 4.1 Requirements File (`requirements.txt`)

```txt
# Core dependencies
openai>=1.104.0
openai-agents>=0.2.10
requests>=2.32.0
httpx>=0.28.0

# Development and testing
pytest>=7.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.0.0

# Utilities
python-dotenv>=1.1.0
pydantic>=2.11.0
click>=8.2.0

# Documentation
sphinx>=7.0.0
sphinx-rtd-theme>=2.0.0
```

#### 4.2 Test Configuration (`tests/test_integration.py`)

```python
import pytest
from src.core.environment import OllamaEnvironment
from src.core.agents import AgentFactory
from src.core.chat import ChatManager
from src.utils.testing import AgentTester

class TestIntegration:
    """Integration tests for complete system"""

    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Setup test environment"""
        try:
            OllamaEnvironment.setup()
            connection = OllamaEnvironment.validate_connection()
            if not connection['connected']:
                pytest.skip("Ollama server not available")
        except Exception:
            pytest.skip("Environment setup failed")

    def test_agent_creation(self):
        """Test all agent types can be created"""
        math_agent = AgentFactory.create_math_tutor()
        coding_agent = AgentFactory.create_coding_assistant()
        writer_agent = AgentFactory.create_creative_writer()

        assert math_agent.name == "Math Tutor"
        assert coding_agent.name == "Coding Assistant"
        assert writer_agent.name == "Creative Writer"

    def test_chat_manager(self):
        """Test chat manager functionality"""
        agents = {
            'math': AgentFactory.create_math_tutor(),
            'coding': AgentFactory.create_coding_assistant()
        }

        chat_manager = ChatManager(agents)

        # Test agent switching
        assert chat_manager.switch_agent('coding')
        assert chat_manager.current_agent_name == 'coding'

        # Test message processing
        result = chat_manager.process_message("Hello")
        assert result.success
        assert result.agent_name == 'coding'

    def test_comprehensive_testing(self):
        """Test the testing framework itself"""
        agents = {
            'math': AgentFactory.create_math_tutor()
        }

        test_scenarios = {
            'math': ["What is 2 + 2?"]
        }

        results = AgentTester.run_comprehensive_tests(agents, test_scenarios)

        assert results.total_tests == 1
        assert results.success_rate >= 0
```

This implementation guide provides a comprehensive, production-ready structure for integrating OpenAI SDK with Ollama. The design emphasizes modularity, testability, and ease of use while maintaining professional code quality standards.