# OpenAI SDK + Ollama API Specification

## Core API Modules

### 1. Environment Configuration Module

#### `setup_ollama_environment()`
**Purpose**: Configure OpenAI SDK to work with local Ollama server

```python
def setup_ollama_environment(
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    disable_tracing: bool = True
) -> bool:
    """
    Configure environment for Ollama compatibility

    Args:
        base_url: Ollama server endpoint
        api_key: Dummy API key for compatibility
        disable_tracing: Disable OpenAI tracing

    Returns:
        bool: Configuration success status

    Raises:
        ConnectionError: If Ollama server is unreachable
        ConfigurationError: If environment setup fails
    """
```

#### `validate_ollama_connection()`
**Purpose**: Test connectivity and model availability

```python
def validate_ollama_connection() -> Dict[str, Any]:
    """
    Validate Ollama server connection and capabilities

    Returns:
        dict: {
            'connected': bool,
            'models': List[str],
            'server_version': str,
            'response_time': float
        }

    Raises:
        ConnectionError: If server is unreachable
        TimeoutError: If connection times out
    """
```

### 2. Agent Factory Module

#### `create_math_tutor()`
**Purpose**: Create specialized mathematics tutor agent

```python
def create_math_tutor(
    model: str = "llama3.2",
    language: str = "auto",
    difficulty_level: str = "adaptive"
) -> Agent:
    """
    Create mathematics tutor agent

    Args:
        model: Ollama model name
        language: Response language ('auto', 'en', 'ko')
        difficulty_level: Problem complexity ('basic', 'intermediate', 'advanced', 'adaptive')

    Returns:
        Agent: Configured math tutor agent

    Features:
        - Step-by-step problem solving
        - Multiple solution methods
        - Concept explanations
        - Practice problem generation
        - Progress tracking
    """
```

#### `create_coding_assistant()`
**Purpose**: Create programming assistance agent

```python
def create_coding_assistant(
    model: str = "llama3.2",
    languages: List[str] = None,
    expertise_level: str = "intermediate"
) -> Agent:
    """
    Create coding assistance agent

    Args:
        model: Ollama model name
        languages: Supported programming languages
        expertise_level: Code complexity handling

    Returns:
        Agent: Configured coding assistant

    Features:
        - Code debugging and optimization
        - Best practices recommendations
        - Multi-language support
        - Algorithm explanations
        - Code review capabilities
    """
```

#### `create_creative_writer()`
**Purpose**: Create creative writing assistance agent

```python
def create_creative_writer(
    model: str = "llama3.2",
    writing_style: str = "adaptive",
    genre_focus: List[str] = None
) -> Agent:
    """
    Create creative writing assistant

    Args:
        model: Ollama model name
        writing_style: Default writing approach
        genre_focus: Preferred writing genres

    Returns:
        Agent: Configured creative writer

    Features:
        - Story development assistance
        - Character creation guidance
        - Plot structure recommendations
        - Style adaptation
        - Creative prompt generation
    """
```

### 3. Conversation Management Module

#### `ChatManager` Class
**Purpose**: Manage interactive conversations with multiple agents

```python
class ChatManager:
    def __init__(self, agents: Dict[str, Agent], default_agent: str = "math"):
        """
        Initialize chat manager with available agents

        Args:
            agents: Dictionary of agent_name -> Agent
            default_agent: Default active agent
        """

    def start_interactive_session(self) -> None:
        """
        Start interactive chat session with command support

        Commands:
            /math - Switch to math tutor
            /coding - Switch to coding assistant
            /creative - Switch to creative writer
            /help - Show available commands
            /quit - Exit session
            /history - Show conversation history
            /clear - Clear conversation context
        """

    def process_message(self, message: str, agent_name: str = None) -> ConversationResult:
        """
        Process single message with specified agent

        Args:
            message: User input message
            agent_name: Target agent (None for current active agent)

        Returns:
            ConversationResult: Response with metadata
        """

    def switch_agent(self, agent_name: str) -> bool:
        """
        Switch to different agent

        Args:
            agent_name: Target agent identifier

        Returns:
            bool: Switch success status
        """
```

### 4. Testing and Validation Module

#### `run_agent_tests()`
**Purpose**: Execute comprehensive agent testing suite

```python
def run_agent_tests(
    agents: Dict[str, Agent] = None,
    test_scenarios: Dict[str, List[str]] = None,
    output_format: str = "detailed"
) -> TestSuiteResult:
    """
    Run comprehensive agent testing

    Args:
        agents: Agents to test (None for all default agents)
        test_scenarios: Custom test cases per agent
        output_format: Result format ('summary', 'detailed', 'json')

    Returns:
        TestSuiteResult: Complete test results with metrics

    Test Categories:
        - Basic functionality tests
        - Error handling tests
        - Performance benchmarks
        - Response quality evaluation
        - Cross-agent consistency tests
    """
```

#### `benchmark_performance()`
**Purpose**: Performance testing and optimization

```python
def benchmark_performance(
    agent: Agent,
    test_queries: List[str],
    iterations: int = 5
) -> PerformanceMetrics:
    """
    Benchmark agent performance

    Args:
        agent: Agent to benchmark
        test_queries: Test input queries
        iterations: Number of test runs per query

    Returns:
        PerformanceMetrics: Detailed performance data

    Metrics:
        - Response time statistics
        - Memory usage patterns
        - Token processing rates
        - Error rates
        - Consistency scores
    """
```

### 5. Utility and Helper Module

#### `get_available_models()`
**Purpose**: Query available Ollama models

```python
def get_available_models() -> List[ModelInfo]:
    """
    Retrieve available Ollama models

    Returns:
        List[ModelInfo]: Available models with metadata

    ModelInfo:
        - name: Model identifier
        - size: Model size in bytes
        - modified: Last modification date
        - digest: Model hash
        - capabilities: Supported features
    """
```

#### `create_custom_agent()`
**Purpose**: Create custom agent with specific configuration

```python
def create_custom_agent(
    name: str,
    instructions: str,
    model: str = "llama3.2",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    tools: List[str] = None
) -> Agent:
    """
    Create custom agent with specific configuration

    Args:
        name: Agent display name
        instructions: System instructions/prompt
        model: Ollama model to use
        temperature: Response creativity (0.0-1.0)
        max_tokens: Maximum response length
        tools: Available tools/functions

    Returns:
        Agent: Configured custom agent
    """
```

## API Usage Examples

### Basic Usage
```python
# Environment setup
setup_ollama_environment()

# Create agents
math_agent = create_math_tutor()
coding_agent = create_coding_assistant()
writer_agent = create_creative_writer()

# Simple interaction
result = Runner.run_sync(math_agent, "Solve x² - 5x + 6 = 0")
print(result.final_output)
```

### Interactive Chat
```python
# Setup chat manager
agents = {
    'math': create_math_tutor(),
    'coding': create_coding_assistant(),
    'creative': create_creative_writer()
}

chat_manager = ChatManager(agents)
chat_manager.start_interactive_session()
```

### Testing and Validation
```python
# Run comprehensive tests
test_results = run_agent_tests()
print(f"Overall success rate: {test_results.success_rate}")

# Performance benchmarking
performance = benchmark_performance(
    agent=math_agent,
    test_queries=["Factor x² + 5x + 6", "Solve 2x + 3 = 7"],
    iterations=10
)
print(f"Average response time: {performance.avg_response_time}ms")
```

### Custom Agent Creation
```python
# Create specialized agent
research_agent = create_custom_agent(
    name="Research Assistant",
    instructions="""You are a research assistant specializing in academic analysis.
    Provide evidence-based responses with proper citations and methodology.""",
    model="llama3.2",
    temperature=0.3
)

# Use custom agent
result = Runner.run_sync(research_agent, "Analyze the impact of AI on education")
```

## Error Handling

### Exception Types
```python
class OllamaConnectionError(Exception):
    """Raised when Ollama server is unreachable"""

class ModelNotFoundError(Exception):
    """Raised when requested model is not available"""

class AgentCreationError(Exception):
    """Raised when agent creation fails"""

class ConversationError(Exception):
    """Raised when conversation processing fails"""
```

### Error Recovery Patterns
```python
try:
    agent = create_math_tutor(model="llama3.2")
    result = Runner.run_sync(agent, user_input)
except ModelNotFoundError:
    # Fallback to different model
    agent = create_math_tutor(model="llama2")
    result = Runner.run_sync(agent, user_input)
except OllamaConnectionError:
    # Provide offline guidance
    result = "Ollama server is not available. Please check connection."
```

## Configuration Options

### Environment Variables
```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_DEFAULT_MODEL=llama3.2
OLLAMA_TIMEOUT=30

# Agent Configuration
AGENT_TEMPERATURE=0.7
AGENT_MAX_TOKENS=2048
AGENT_LANGUAGE=auto
```

### Configuration File Support
```python
# config.yaml
ollama:
  base_url: "http://localhost:11434/v1"
  default_model: "llama3.2"
  timeout: 30

agents:
  math:
    temperature: 0.3
    language: "auto"
  coding:
    temperature: 0.5
    max_tokens: 4096
  creative:
    temperature: 0.8
    style: "adaptive"
```

## Response Data Structures

### ConversationResult
```python
@dataclass
class ConversationResult:
    agent_name: str
    user_input: str
    final_output: str
    execution_time: float
    success: bool
    metadata: Dict[str, Any]
    error: Optional[str] = None
```

### TestSuiteResult
```python
@dataclass
class TestSuiteResult:
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    execution_time: float
    detailed_results: List[TestResult]
    summary: str
```

### PerformanceMetrics
```python
@dataclass
class PerformanceMetrics:
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    memory_usage_mb: float
    tokens_per_second: float
    error_rate: float
    consistency_score: float
```