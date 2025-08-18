#!/usr/bin/env python3
"""
Grok-3-Mini Comprehensive Test
Tests grok-3-mini performance for RAG tasks
"""

import asyncio
import os
from lightrag.llm.xai import xai_complete_if_cache


async def test_grok_3_mini_capabilities():
    """Test grok-3-mini with various RAG-style tasks"""
    
    print("🚀 Testing Grok-3-Mini Capabilities")
    print("=" * 40)
    
    test_cases = [
        {
            "name": "Entity Extraction",
            "system": "You are an expert at extracting entities from text. Extract all entities and return them as a JSON list.",
            "prompt": "LightRAG is a Python-based retrieval system developed by Hong Kong University that uses knowledge graphs and vector search for document processing.",
            "expected_type": "JSON with entities"
        },
        {
            "name": "Relationship Extraction", 
            "system": "Extract relationships between entities in the text. Return as JSON with source, relation, target.",
            "prompt": "Microsoft developed Azure OpenAI. OpenAI created GPT-4. Azure OpenAI provides GPT-4 models.",
            "expected_type": "JSON with relationships"
        },
        {
            "name": "Summarization",
            "system": "Provide a concise summary of the following text.",
            "prompt": "LightRAG combines traditional vector search with knowledge graph technology to improve retrieval accuracy. The system processes documents by extracting entities and relationships, then stores them in both vector databases and graph structures. This hybrid approach allows for more contextual and accurate responses to user queries.",
            "expected_type": "Summary paragraph"
        },
        {
            "name": "Question Answering",
            "system": "Answer the question based on the provided context accurately and concisely.",
            "prompt": "Context: LightRAG supports multiple storage backends including PostgreSQL, Redis, Neo4j, and file-based storage.\nQuestion: What storage options does LightRAG support?",
            "expected_type": "Direct answer"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/4: {test['name']}")
        print(f"Expected: {test['expected_type']}")
        
        try:
            response = await xai_complete_if_cache(
                model=os.getenv('LLM_MODEL', 'grok-3-mini'),
                prompt=test['prompt'],
                system_prompt=test['system'],
                api_key=os.getenv('LLM_BINDING_API_KEY'),
                base_url=os.getenv('LLM_BINDING_HOST', 'https://api.x.ai/v1'),
                temperature=0.1,
                max_tokens=500
            )
            
            print(f"✅ Response: {response[:150]}...")
            if len(response) > 150:
                print(f"          ... (truncated, full length: {len(response)} chars)")
            results.append(True)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(False)
    
    # Performance summary
    print(f"\n📊 Grok-3-Mini Performance Summary")
    print("=" * 40)
    print(f"✅ Successful tasks: {sum(results)}/{len(results)}")
    print(f"🎯 Success rate: {sum(results)/len(results)*100:.1f}%")
    
    if all(results):
        print("\n🎉 Grok-3-Mini is ready for LightRAG!")
        print("✅ All RAG-style tasks completed successfully")
    else:
        print(f"\n⚠️ Some tasks failed - check model capabilities")
    
    return all(results)


def main():
    print("🧪 Grok-3-Mini Comprehensive Test")
    return asyncio.run(test_grok_3_mini_capabilities())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)