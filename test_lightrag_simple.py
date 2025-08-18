#!/usr/bin/env python3
"""
Simple LightRAG Component Test
Tests individual components without full pipeline
"""

import asyncio
import os
from lightrag.utils import EmbeddingFunc
from lightrag.llm.xai import xai_complete_if_cache
from lightrag.llm.ollama import ollama_embed


async def test_xai_llm():
    """Test xAI LLM connection"""
    print("\n🤖 Testing xAI LLM...")
    
    try:
        response = await xai_complete_if_cache(
            model=os.getenv('LLM_MODEL', 'grok-beta'),
            prompt="Hello! Please respond with exactly: 'xAI connection successful'",
            system_prompt="You are a helpful assistant. Always respond exactly as requested.",
            api_key=os.getenv('LLM_BINDING_API_KEY'),
            base_url=os.getenv('LLM_BINDING_HOST', 'https://api.x.ai'),
            temperature=0.0,
        )
        print(f"✅ xAI Response: {response}")
        return True
    except Exception as e:
        print(f"❌ xAI Error: {e}")
        return False


async def test_ollama_embedding():
    """Test Ollama embedding function"""
    print("\n🔢 Testing Ollama Embeddings...")
    
    try:
        test_texts = ["Hello world", "LightRAG is working"]
        embeddings = await ollama_embed(
            texts=test_texts,
            embed_model=os.getenv('EMBEDDING_MODEL', 'bge-m3:latest'),
            host=os.getenv('EMBEDDING_BINDING_HOST', 'http://localhost:11434'),
        )
        print(f"✅ Ollama Embeddings: {embeddings.shape} shape")
        print(f"    Sample values: {embeddings[0][:5]}...")
        return True
    except Exception as e:
        print(f"❌ Ollama Error: {e}")
        return False


async def test_embedding_func():
    """Test EmbeddingFunc wrapper"""
    print("\n🔧 Testing EmbeddingFunc wrapper...")
    
    try:
        embedding_func = EmbeddingFunc(
            embedding_dim=int(os.getenv('EMBEDDING_DIM', '1024')),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv('EMBEDDING_MODEL', 'bge-m3:latest'),
                host=os.getenv('EMBEDDING_BINDING_HOST', 'http://localhost:11434'),
            )
        )
        
        # Test the wrapper
        test_texts = ["This is a test"]
        result = await embedding_func(test_texts)
        print(f"✅ EmbeddingFunc: {result.shape} shape, dim={embedding_func.embedding_dim}")
        return True
    except Exception as e:
        print(f"❌ EmbeddingFunc Error: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Simple LightRAG Component Test")
    print("=" * 45)
    
    # Check configuration
    required_vars = ['LLM_BINDING_API_KEY', 'LLM_MODEL', 'EMBEDDING_MODEL']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var, '')
        if not value or value.endswith('_here'):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing configuration: {', '.join(missing_vars)}")
        return
    
    print("✅ Configuration looks good!")
    
    # Run tests
    async def run_tests():
        results = []
        results.append(await test_xai_llm())
        results.append(await test_ollama_embedding())
        results.append(await test_embedding_func())
        
        print(f"\n📊 Test Results:")
        print(f"✅ Passed: {sum(results)}/{len(results)}")
        
        if all(results):
            print("\n🎉 All component tests passed!")
            print("✅ xAI + Ollama setup is working correctly")
            return True
        else:
            print("\n⚠️ Some tests failed - check configuration")
            return False
    
    return asyncio.run(run_tests())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)