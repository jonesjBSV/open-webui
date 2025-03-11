from open_webui.config import VECTOR_DB

if VECTOR_DB == "milvus":
    from open_webui.retrieval.vector.dbs.milvus import MilvusClient

    VECTOR_DB_CLIENT = MilvusClient()
elif VECTOR_DB == "qdrant":
    from open_webui.retrieval.vector.dbs.qdrant import QdrantClient

    VECTOR_DB_CLIENT = QdrantClient()
elif VECTOR_DB == "opensearch":
    from open_webui.retrieval.vector.dbs.opensearch import OpenSearchClient

    VECTOR_DB_CLIENT = OpenSearchClient()
elif VECTOR_DB == "pgvector":
    from open_webui.retrieval.vector.dbs.pgvector import PgvectorClient

    VECTOR_DB_CLIENT = PgvectorClient()
elif VECTOR_DB == "weaviate":
    try:
        from open_webui.retrieval.vector.dbs.weaviate import WeaviateClient
        VECTOR_DB_CLIENT = WeaviateClient()
        print("Weaviate client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Weaviate client: {e}")
        print("Falling back to default vector database client.")
        from open_webui.retrieval.vector.dbs.chroma import ChromaClient
        VECTOR_DB_CLIENT = ChromaClient()
else:
    from open_webui.retrieval.vector.dbs.chroma import ChromaClient

    VECTOR_DB_CLIENT = ChromaClient()
