#!/usr/bin/env python
"""
Test script for GraphRAG functionality with Weaviate in Open WebUI.

This script demonstrates how to:
1. Create sample documents
2. Upload them to Weaviate with graph relationships
3. Query using graph-enhanced search

This script should be run inside the Open WebUI Docker container
where all dependencies are already installed.

To run this script:
1. Copy it to the container: docker cp test_graphrag.py open-webui:/app/backend/
2. Execute it in the container: docker exec -it open-webui python /app/backend/test_graphrag.py
"""
"""

import os
import json
import uuid
import requests
from typing import List, Dict, Any

# Base URL for Open WebUI API
BASE_URL = "http://localhost:8000"  # Change this if your server runs on a different port

# Sample text documents for testing
SAMPLE_DOCS = [
    {
        "title": "Introduction to Vector Databases",
        "content": """
# Introduction to Vector Databases

Vector databases are specialized database systems designed to store, manage, and search high-dimensional vectors that represent embeddings of text, images, audio, or other data.

## Key Features

- Efficient similarity search
- Support for high-dimensional vectors
- Optimized for machine learning applications
- Fast retrieval of semantically similar items
        """
    },
    {
        "title": "Understanding GraphRAG",
        "content": """
# Understanding GraphRAG

GraphRAG (Graph-based Retrieval Augmented Generation) enhances traditional RAG systems by incorporating graph relationships between document chunks.

## Benefits of GraphRAG

- Preserves document structure
- Captures semantic relationships between chunks
- Improves context for generation
- Enables more sophisticated retrieval strategies

### Types of Relationships in GraphRAG

1. **Semantic relationships**: Connections based on meaning
2. **Hierarchical relationships**: Parent-child structures
3. **Sequential relationships**: Preserving document flow
        """
    }
]

def create_test_files() -> List[str]:
    """Create test files and upload them to the server"""
    file_ids = []
    
    for doc in SAMPLE_DOCS:
        # Create a temporary file
        temp_file_path = f"/tmp/{uuid.uuid4()}.md"
        with open(temp_file_path, "w") as f:
            f.write(doc["content"])
        
        # Upload file to server
        print(f"Uploading test file: {doc['title']}")
        with open(temp_file_path, "rb") as f:
            files = {"file": (f"{doc['title']}.md", f, "text/markdown")}
            response = requests.post(
                f"{BASE_URL}/files/upload",
                files=files
            )
            
            if response.status_code == 200:
                file_data = response.json()
                file_id = file_data.get("id")
                file_ids.append(file_id)
                print(f"Successfully uploaded file with ID: {file_id}")
            else:
                print(f"Error uploading file: {response.text}")
        
        # Clean up temporary file
        os.remove(temp_file_path)
    
    return file_ids

def test_graphrag_upload(file_ids: List[str]) -> bool:
    """Test uploading files to Weaviate with GraphRAG relationships"""
    if not file_ids:
        print("No files to test with!")
        return False
    
    # Create a collection for our test
    collection_name = f"test_graphrag_{uuid.uuid4().hex[:8]}"
    
    # Upload files for GraphRAG processing
    print(f"\nProcessing files for GraphRAG in collection: {collection_name}")
    payload = {
        "file_ids": file_ids,
        "collection_name": collection_name,
        "relationship_type": "semantic",  # Options: semantic, hierarchical, sequential
        "chunk_size": 200,
        "chunk_overlap": 50
    }
    
    response = requests.post(
        f"{BASE_URL}/retrieval/process/graphrag",
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"GraphRAG processing successful!")
        print(f"Collection name: {result['collection_name']}")
        print(f"Relationship type: {result['relationship_type']}")
        print(f"Graph relationships created: {result['graph_relationships_created']}")
        
        # Print details about processed chunks
        for res in result['results']:
            print(f"File {res['file_id']} processed with status: {res['status']}")
            if res.get('chunk_ids'):
                print(f"  Number of chunks: {len(res['chunk_ids'])}")
        
        return True
    else:
        print(f"Error processing files for GraphRAG: {response.text}")
        return False

def test_graphrag_search(collection_name: str, query: str) -> None:
    """Test searching with GraphRAG enhanced retrieval"""
    print(f"\nTesting GraphRAG search in collection: {collection_name}")
    print(f"Query: '{query}'")
    
    # First, get an embedding for our query
    embed_response = requests.post(
        f"{BASE_URL}/retrieval/embedding",
        json={"text": query}
    )
    
    if embed_response.status_code != 200:
        print(f"Error getting embedding: {embed_response.text}")
        return
    
    query_embedding = embed_response.json()["embedding"]
    
    # Now use our custom graph_search endpoint (assumes it's been implemented in the API)
    search_payload = {
        "collection_name": collection_name,
        "query_vector": query_embedding,
        "limit": 5,
        "depth": 1  # How many hops in the graph to traverse
    }
    
    search_response = requests.post(
        f"{BASE_URL}/retrieval/query/graph",
        json=search_payload
    )
    
    if search_response.status_code == 200:
        results = search_response.json()
        print(f"Retrieved {len(results['documents'])} results:")
        
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"\nResult {i+1}:")
            print(f"Document: {doc[:100]}...")  # Show first 100 chars
            print(f"Is direct match: {metadata.get('is_direct_match', False)}")
            print(f"Chunk type: {metadata.get('chunk_type', 'unknown')}")
            print(f"Distance: {results['distances'][i]}")
    else:
        print(f"Error during search: {search_response.text}")

def main():
    """Main test function"""
    print("=== Testing GraphRAG Implementation ===\n")
    
    # Step 1: Create and upload test files
    file_ids = create_test_files()
    if not file_ids:
        print("Failed to create test files")
        return
    
    # Step 2: Test GraphRAG upload
    collection_name = f"test_graphrag_{uuid.uuid4().hex[:8]}"
    success = test_graphrag_upload(file_ids)
    if not success:
        print("GraphRAG upload test failed")
        return
    
    # Step 3: Test GraphRAG search
    test_queries = [
        "How do vector databases work?",
        "What are the benefits of GraphRAG?",
        "Explain hierarchical relationships in document retrieval"
    ]
    
    for query in test_queries:
        test_graphrag_search(collection_name, query)
    
    print("\n=== GraphRAG Testing Complete ===")

if __name__ == "__main__":
    main()
