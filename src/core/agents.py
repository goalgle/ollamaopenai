from typing import Optional, List, Dict, Any
from .environment import OllamaEnvironment

try:
    from agents import Agent
except ImportError:
    # Fallback for testing without agents package
    class Agent:
        def __init__(self, name, instructions, model, **kwargs):
            self.name = name
            self.instructions = instructions
            self.model = model
            # Accept all keyword arguments for compatibility
            for key, value in kwargs.items():
                setattr(self, key, value)

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

        try:
            # Try with model_settings for real agents package
            from agents import ModelSettings
            model_settings = ModelSettings(temperature=temperature)
            return Agent(
                name="Math Tutor",
                instructions=instructions,
                model=model,
                model_settings=model_settings
            )
        except ImportError:
            # Fallback for mock/test scenarios
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

        try:
            # Try with model_settings for real agents package
            from agents import ModelSettings
            model_settings = ModelSettings(temperature=temperature)
            return Agent(
                name="Coding Assistant",
                instructions=instructions,
                model=model,
                model_settings=model_settings
            )
        except ImportError:
            # Fallback for mock/test scenarios
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

        try:
            # Try with model_settings for real agents package
            from agents import ModelSettings
            model_settings = ModelSettings(temperature=temperature)
            return Agent(
                name="Creative Writer",
                instructions=instructions,
                model=model,
                model_settings=model_settings
            )
        except ImportError:
            # Fallback for mock/test scenarios
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

        try:
            # Try with model_settings for real agents package
            from agents import ModelSettings
            model_settings = ModelSettings(temperature=temperature)
            return Agent(
                name=name,
                instructions=instructions,
                model=model,
                model_settings=model_settings,
                **kwargs
            )
        except ImportError:
            # Fallback for mock/test scenarios
            return Agent(
                name=name,
                instructions=instructions,
                model=model,
                temperature=temperature,
                **kwargs
            )