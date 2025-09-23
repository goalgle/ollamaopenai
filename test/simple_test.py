#!/usr/bin/env python3
"""
Simple test script for OpenAI SDK + Ollama integration
Validates basic functionality and provides quick validation
"""

import os
import sys
import time
import requests
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_ollama_connection():
    """Test basic Ollama server connectivity"""
    print("🔍 Testing Ollama connection...")

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()

        data = response.json()
        models = [m['name'] for m in data.get('models', [])]

        print(f"✅ Ollama server is running")
        print(f"📋 Available models: {models}")

        if not models:
            print("⚠️  No models found. Please download a model:")
            print("   ollama pull llama3.2")
            return False

        return True

    except requests.RequestException as e:
        print(f"❌ Ollama connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Start Ollama server: ollama serve")
        print("2. Download a model: ollama pull llama3.2")
        print("3. Check if port 11434 is available")
        return False

def test_environment_setup():
    """Test environment configuration"""
    print("\n🔧 Testing environment setup...")

    try:
        # Import after path setup
        from core.environment import OllamaEnvironment

        # Setup environment
        success = OllamaEnvironment.setup()

        if success:
            print("✅ Environment setup successful")

            # Validate configuration
            connection_info = OllamaEnvironment.validate_connection()

            if connection_info['connected']:
                print(f"✅ Connection validated")
                print(f"📊 Server status: {connection_info['server_status']}")
                return True
            else:
                print(f"❌ Connection validation failed")
                return False
        else:
            print("❌ Environment setup failed")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the src directory structure is correct")
        return False
    except Exception as e:
        print(f"❌ Environment setup error: {e}")
        return False

def test_agent_creation():
    """Test agent creation and basic functionality"""
    print("\n🤖 Testing agent creation...")

    try:
        from core.agents import AgentFactory
        from agents import Runner

        # Create a simple math agent
        print("   Creating math tutor agent...")
        math_agent = AgentFactory.create_math_tutor()
        print(f"✅ Math agent created: {math_agent.name}")

        # Test simple interaction
        print("   Testing basic interaction...")
        test_query = "What is 2 + 2?"

        start_time = time.time()
        result = Runner.run_sync(math_agent, test_query)
        end_time = time.time()

        response_time = end_time - start_time
        response = result.final_output

        print(f"✅ Response received in {response_time:.2f}s")
        print(f"📝 Query: {test_query}")
        print(f"🤖 Response: {response[:100]}...")

        # Basic validation
        if len(response) > 0:
            print("✅ Response validation passed")
            return True
        else:
            print("❌ Empty response received")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Agent creation/testing error: {e}")
        return False

def test_multiple_agents():
    """Test creation and basic functionality of multiple agent types"""
    print("\n🔄 Testing multiple agent types...")

    try:
        from core.agents import AgentFactory
        from agents import Runner

        # Test data for each agent type
        agent_tests = [
            ("math", "create_math_tutor", "Calculate 5 * 7"),
            ("coding", "create_coding_assistant", "Write a hello world in Python"),
            ("creative", "create_creative_writer", "Write a short poem about AI")
        ]

        results = {}

        for agent_type, factory_method, test_query in agent_tests:
            print(f"   Testing {agent_type} agent...")

            try:
                # Create agent
                factory = getattr(AgentFactory, factory_method)
                agent = factory()

                # Test interaction
                start_time = time.time()
                result = Runner.run_sync(agent, test_query)
                end_time = time.time()

                response_time = end_time - start_time
                response_length = len(result.final_output)

                results[agent_type] = {
                    'success': True,
                    'response_time': response_time,
                    'response_length': response_length
                }

                print(f"   ✅ {agent_type} agent: {response_time:.2f}s, {response_length} chars")

            except Exception as e:
                results[agent_type] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"   ❌ {agent_type} agent failed: {e}")

        # Summary
        successful_agents = [k for k, v in results.items() if v.get('success', False)]
        print(f"\n📊 Agent Test Summary:")
        print(f"   Successful: {len(successful_agents)}/{len(agent_tests)}")
        print(f"   Working agents: {', '.join(successful_agents)}")

        return len(successful_agents) > 0

    except Exception as e:
        print(f"❌ Multiple agent testing error: {e}")
        return False

def test_chat_manager():
    """Test chat manager functionality"""
    print("\n💬 Testing chat manager...")

    try:
        from core.agents import AgentFactory
        from core.chat import ChatManager

        # Create agents
        agents = {
            'math': AgentFactory.create_math_tutor(),
            'coding': AgentFactory.create_coding_assistant()
        }

        # Create chat manager
        chat_manager = ChatManager(agents, default_agent='math')
        print("✅ Chat manager created")

        # Test agent switching
        switch_success = chat_manager.switch_agent('coding')
        if switch_success and chat_manager.current_agent_name == 'coding':
            print("✅ Agent switching works")
        else:
            print("❌ Agent switching failed")
            return False

        # Test message processing
        result = chat_manager.process_message("Hello, can you help me?")

        if result.success:
            print(f"✅ Message processing works ({result.execution_time:.2f}s)")
            return True
        else:
            print(f"❌ Message processing failed: {result.error}")
            return False

    except Exception as e:
        print(f"❌ Chat manager testing error: {e}")
        return False

def run_all_tests():
    """Run all tests in sequence"""
    print("🧪 OpenAI SDK + Ollama Simple Test Suite")
    print("=" * 60)

    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Environment Setup", test_environment_setup),
        ("Agent Creation", test_agent_creation),
        ("Multiple Agents", test_multiple_agents),
        ("Chat Manager", test_chat_manager)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Final summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed_tests = []
    failed_tests = []

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

        if success:
            passed_tests.append(test_name)
        else:
            failed_tests.append(test_name)

    print(f"\nResults: {len(passed_tests)}/{len(tests)} tests passed")

    if len(passed_tests) == len(tests):
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        if failed_tests:
            print(f"Failed tests: {', '.join(failed_tests)}")
        return False

def main():
    """Main test execution"""
    success = run_all_tests()

    if success:
        print("\n🚀 Quick Start:")
        print("python examples/basic_usage.py")
        print("python examples/interactive_chat.py")
        sys.exit(0)
    else:
        print("\n🔧 Please fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()