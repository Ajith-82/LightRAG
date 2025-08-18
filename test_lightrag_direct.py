#!/usr/bin/env python3
"""
Direct LightRAG Test Script
Tests LightRAG functionality without the API server
"""

import asyncio
import os
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.xai import xai_complete_if_cache, xai_embed  
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status


async def test_lightrag_basic():
    """Test basic LightRAG functionality with xAI + Ollama"""
    
    print("🚀 Testing LightRAG with xAI (LLM) + Ollama (Embeddings)")
    print("=" * 60)
    
    # Create working directory
    working_dir = Path("./rag_storage_test")
    working_dir.mkdir(exist_ok=True)
    
    try:
        # Create LLM and embedding functions
        print("\n⚙️ Creating LLM and embedding functions...")
        
        # xAI LLM function
        async def xai_model_complete(
            prompt, system_prompt=None, history_messages=None, **kwargs
        ):
            if history_messages is None:
                history_messages = []
            temperature = os.getenv('TEMPERATURE', '0')
            kwargs["temperature"] = float(temperature)
            
            # Ensure hashing_kv is available if provided
            hashing_kv = kwargs.pop("hashing_kv", None)
            
            return await xai_complete_if_cache(
                os.getenv('LLM_MODEL', 'grok-beta'),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=os.getenv('LLM_BINDING_API_KEY'),
                base_url=os.getenv('LLM_BINDING_HOST', 'https://api.x.ai'),
                hashing_kv=hashing_kv,
                **kwargs,
            )
        
        # Ollama embedding function with proper wrapper
        embedding_func = EmbeddingFunc(
            embedding_dim=int(os.getenv('EMBEDDING_DIM', '1024')),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv('EMBEDDING_MODEL', 'bge-m3:latest'),
                host=os.getenv('EMBEDDING_BINDING_HOST', 'http://localhost:11434'),
            )
        )
        
        # Initialize LightRAG with proper functions
        print("\n📝 Initializing LightRAG...")
        rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=xai_model_complete,
            embedding_func=embedding_func,
        )
        
        # Initialize storage systems
        print("📊 Initializing storage backends...")
        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        # Test document insertion
        print("\n📄 Testing document insertion...")
        test_content = """
        LightRAG is a powerful retrieval-augmented generation system that combines 
        knowledge graphs with vector search capabilities. It supports multiple LLM 
        providers including OpenAI, xAI, and Ollama. The system can process various 
        document formats and extract entities and relationships to build comprehensive 
        knowledge graphs for enhanced information retrieval.
        
        Key features include:
        - Multi-modal document processing
        - Knowledge graph construction
        - Vector similarity search
        - Hybrid retrieval modes
        - Production-ready deployment options
        """
        
        result = await rag.ainsert(test_content)
        print(f"✅ Document inserted successfully: {result}")
        
        # Test different query modes
        test_queries = [
            ("What is LightRAG?", "local"),
            ("What are the key features?", "global"), 
            ("How does it work with knowledge graphs?", "hybrid"),
        ]
        
        print("\n🔍 Testing different query modes...")
        for query, mode in test_queries:
            print(f"\n📋 Query: '{query}' (mode: {mode})")
            try:
                param = QueryParam(mode=mode, top_k=5)
                response = await rag.aquery(query, param=param)
                print(f"✅ Response: {response[:200]}...")
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        print("\n🎯 Basic LightRAG test completed!")
        
    except Exception as e:
        print(f"❌ Error during LightRAG test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'rag' in locals():
            try:
                await rag.finalize_storages()
                print("✅ Storage cleanup completed")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")


async def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing Configuration...")
    print("=" * 30)
    
    # Check environment variables
    env_vars = [
        'LLM_BINDING', 'LLM_MODEL', 'LLM_BINDING_API_KEY',
        'EMBEDDING_BINDING', 'EMBEDDING_MODEL', 'EMBEDDING_BINDING_HOST'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        # Hide API keys for security
        if 'API_KEY' in var and value != 'Not set':
            value = f"{value[:8]}..." if len(value) > 8 else "***"
        print(f"  {var}: {value}")


def main():
    """Main test function"""
    print("🧪 LightRAG Direct Test")
    print("=" * 40)
    
    # Check if we need API keys
    if os.getenv('LLM_BINDING_API_KEY', '').endswith('_here'):
        print("\n⚠️  WARNING: Please update your xAI API key in .env file")
        print("   LLM_BINDING_API_KEY=your_actual_xai_api_key")
        print("\n🔄 You can still test with mock responses by setting:")
        print("   LLM_BINDING_API_KEY=test_key")
        return
    
    # Run async tests
    asyncio.run(test_configuration())
    asyncio.run(test_lightrag_basic())


if __name__ == "__main__":
    main()