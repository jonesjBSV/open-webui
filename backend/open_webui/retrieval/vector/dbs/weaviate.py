import weaviate
import json
from typing import Optional, List, Dict, Any, Union

from open_webui.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import (
    WEAVIATE_URL,
    WEAVIATE_API_KEY,
    WEAVIATE_GRPC_URL,
    WEAVIATE_BATCH_SIZE,
    WEAVIATE_TENANT,
)


class WeaviateClient:
    def __init__(self):
        try:
            # Initialize client configuration for Weaviate 3.x
            auth_config = None
            if WEAVIATE_API_KEY:
                # AuthClientCredentials was used in 3.x instead of AuthApiKey
                auth_config = weaviate.auth.AuthClientCredentials(api_key=WEAVIATE_API_KEY)
            
            # Headers for multi-tenancy if needed
            headers = None
            if WEAVIATE_TENANT:
                headers = {"X-Weaviate-Tenant": WEAVIATE_TENANT}
            
            # For Weaviate client 3.x, the initialization is simpler
            self.client = weaviate.Client(
                url=WEAVIATE_URL,
                auth_client_secret=auth_config,
                additional_headers=headers
            )
            
            self.batch_size = WEAVIATE_BATCH_SIZE or 100
            
            # Test connection and configure batch settings for 3.x
            self.client.schema.get()
            
            # Configure batch settings for 3.x version
            self.batch = self.client.batch
            
            print(f"Weaviate client v3.x initialized successfully with URL: {WEAVIATE_URL}")
            
        except Exception as e:
            print(f"Error initializing Weaviate client: {e}")
            # Create a placeholder client that won't throw errors
            self.client = None
            self.batch_size = 100

    def has_collection(self, collection_name: str) -> bool:
        # Check if the class exists based on the collection name
        if not self.client:
            return False
        try:
            return self.client.schema.exists(collection_name)
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False

    def delete_collection(self, collection_name: str):
        # Delete the class based on the collection name
        if not self.client:
            return False
        try:
            if self.has_collection(collection_name):
                self.client.schema.delete_class(collection_name)
                return True
            return False
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False

    def _create_schema(self, collection_name: str):
        # Create schema with properties for graphRAG
        # Define properties directly as a dictionary structure
        properties = [
            {
                "name": "text",
                "dataType": ["text"],
            },
            {
                "name": "chunk_id",
                "dataType": ["text"],
            },
            {
                "name": "file_id",
                "dataType": ["text"],
            },
            {
                "name": "file_name",
                "dataType": ["text"],
            },
            {
                "name": "file_type",
                "dataType": ["text"],
            },
            {
                "name": "page_number",
                "dataType": ["int"],
            },
            {
                "name": "total_pages",
                "dataType": ["int"],
            },
            {
                "name": "source_url",
                "dataType": ["text"],
            },
            {
                "name": "source_title",
                "dataType": ["text"],
            },
            # GraphRAG custom properties for relationships
            {
                "name": "next_chunk",
                "dataType": ["text"],
            },
            {
                "name": "prev_chunk",
                "dataType": ["text"],
            },
            {
                "name": "parent_chunk",
                "dataType": ["text"],
            },
            {
                "name": "child_chunks",
                "dataType": ["text[]"]
            },
            {
                "name": "related_chunks",
                "dataType": ["text[]"]
            },
        ]
        
        class_obj = {
            "class": collection_name,
            "vectorizer": "none",  # We'll provide vectors explicitly
            "properties": properties,
            "vectorIndexConfig": {
                "distance": "cosine",
            }
        }
        
        try:
            self.client.schema.create_class(class_obj)
        except Exception as e:
            # Log error but don't raise - we'll handle it gracefully
            print(f"Error creating schema: {e}")

    def search(
        self, collection_name: str, vectors: list[list[float]], limit: int
    ) -> Optional[SearchResult]:
        # Search for the nearest neighbor items based on the vectors and return 'limit' number of results
        try:
            if not self.has_collection(collection_name):
                return None

            # Using vector search
            query = (self.client.query
                .get(collection_name, ["id", "text", "_additional {distance}"])
                .with_near_vector({"vector": vectors[0]})
                .with_limit(limit))
            
            result = query.do()
            
            if not result or "data" not in result or "Get" not in result["data"] or collection_name not in result["data"]["Get"]:
                return None
            
            items = result["data"]["Get"][collection_name]
            
            # Extract results in the format expected by Open WebUI
            ids = [[item["id"]] for item in items]
            distances = [[item["_additional"]["distance"]] for item in items]
            documents = [[item["text"]] for item in items]
            
            # Get metadata for each item
            metadatas = []
            for item_id in [item["id"] for item in items]:
                metadata_result = (self.client.query
                    .get(collection_name, ["chunk_id", "file_id", "file_name", "file_type", "page_number", 
                                         "total_pages", "source_url", "source_title", "next_chunk", 
                                         "prev_chunk", "parent_chunk", "child_chunks", "related_chunks"])
                    .with_where({"path": ["id"], "operator": "Equal", "valueString": item_id})
                    .do())
                
                if metadata_result and "data" in metadata_result and "Get" in metadata_result["data"] and collection_name in metadata_result["data"]["Get"]:
                    meta_item = metadata_result["data"]["Get"][collection_name][0]
                    # Remove None values
                    metadata = {k: v for k, v in meta_item.items() if v is not None}
                    metadatas.append([metadata])
                else:
                    metadatas.append([{}])
            
            return SearchResult(
                **{
                    "ids": ids,
                    "distances": distances,
                    "documents": documents,
                    "metadatas": metadatas,
                }
            )
        except Exception as e:
            print(f"Error in vector search: {e}")
            return None

    def query(
        self, collection_name: str, filter: dict, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        # Query the items from the collection based on the filter
        try:
            if not self.has_collection(collection_name):
                return None
            
            # Convert the filter to Weaviate where clause
            where_clause = self._convert_filter_to_where(filter)
            
            # Query Weaviate
            properties = ["id", "text", "chunk_id", "file_id", "file_name", "file_type", 
                         "page_number", "total_pages", "source_url", "source_title", 
                         "next_chunk", "prev_chunk", "parent_chunk", "child_chunks", "related_chunks"]
            
            query = (self.client.query
                    .get(collection_name, properties)
                    .with_where(where_clause))
            
            if limit:
                query = query.with_limit(limit)
            
            result = query.do()
            
            if not result or "data" not in result or "Get" not in result["data"] or collection_name not in result["data"]["Get"]:
                return None
            
            items = result["data"]["Get"][collection_name]
            
            # Extract the results in the format expected by Open WebUI
            ids = [[item["id"] for item in items]]
            documents = [[item["text"] for item in items]]
            
            # Extract metadata
            metadatas = [[{k: v for k, v in item.items() if k != "id" and k != "text" and v is not None} for item in items]]
            
            return GetResult(
                **{
                    "ids": ids,
                    "documents": documents,
                    "metadatas": metadatas,
                }
            )
        except Exception as e:
            print(f"Error in query: {e}")
            return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        # Get all the items in the collection
        try:
            if not self.has_collection(collection_name):
                return None
                
            properties = ["id", "text", "chunk_id", "file_id", "file_name", "file_type", 
                         "page_number", "total_pages", "source_url", "source_title", 
                         "next_chunk", "prev_chunk", "parent_chunk", "child_chunks", "related_chunks"]
            
            result = self.client.query.get(collection_name, properties).do()
            
            if not result or "data" not in result or "Get" not in result["data"] or collection_name not in result["data"]["Get"]:
                return None
            
            items = result["data"]["Get"][collection_name]
            
            # Extract the results in the format expected by Open WebUI
            ids = [[item["id"] for item in items]]
            documents = [[item["text"] for item in items]]
            
            # Extract metadata
            metadatas = [[{k: v for k, v in item.items() if k != "id" and k != "text" and v is not None} for item in items]]
            
            return GetResult(
                **{
                    "ids": ids,
                    "documents": documents,
                    "metadatas": metadatas,
                }
            )
        except Exception as e:
            print(f"Error in get: {e}")
            return None

    def insert(self, collection_name: str, items: list[VectorItem]):
        # Insert the items into the collection; if the collection doesn't exist, create it
        if not self.has_collection(collection_name):
            self._create_schema(collection_name)
        
        try:
            # In Weaviate 3.x, batch import works slightly differently
            # Configure batch size
            self.batch.configure(batch_size=self.batch_size)
            
            # Begin batch process
            with self.batch as batch:
                for item in items:
                    # Prepare the object data
                    properties = {"text": item["text"]}
                    
                    # Add metadata properties
                    if "metadata" in item and item["metadata"]:
                        for key, value in item["metadata"].items():
                            if value is not None:
                                properties[key] = value
                    
                    # Add the object with its vector - syntax is compatible with 3.x
                    batch.add_data_object(
                        data_object=properties,
                        class_name=collection_name,
                        uuid=item["id"],
                        vector=item["vector"]
                    )
        except Exception as e:
            print(f"Error in insert: {e}")

    def upsert(self, collection_name: str, items: list[VectorItem]):
        # Upsert items in the collection; if collection doesn't exist, create it
        if not self.has_collection(collection_name):
            self._create_schema(collection_name)
        
        try:
            # In Weaviate 3.x, batch import works slightly differently
            # Configure batch size
            self.batch.configure(batch_size=self.batch_size)
            
            # Begin batch process
            with self.batch as batch:
                for item in items:
                    # Prepare the object data
                    properties = {"text": item["text"]}
                    
                    # Add metadata properties
                    if "metadata" in item and item["metadata"]:
                        for key, value in item["metadata"].items():
                            if value is not None:
                                properties[key] = value
                    
                    # For Weaviate 3.x upsert behavior
                    # In version 3.x, add_data_object will update the object if it exists or create it if it doesn't
                    # We don't need to explicitly check for existence
                    batch.add_data_object(
                        data_object=properties,
                        class_name=collection_name,
                        uuid=item["id"],
                        vector=item["vector"]
                    )
        except Exception as e:
            print(f"Error in upsert: {e}")

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        # Delete items from the collection based on ids or filter
        try:
            if not self.has_collection(collection_name):
                print(f"Collection {collection_name} does not exist. Nothing to delete.")
                return
            
            if ids:
                # Delete by IDs - batch delete for better performance in version 3.x
                try:
                    # Configure batch size for deletion
                    self.batch.configure(batch_size=self.batch_size)
                    
                    # Use batch for deletions in 3.x
                    with self.batch as batch:
                        for item_id in ids:
                            batch.delete_objects(
                                class_name=collection_name,
                                where={
                                    "path": ["id"],
                                    "operator": "Equal",
                                    "valueString": item_id
                                }
                            )
                except Exception as e:
                    print(f"Error in batch delete by IDs: {e}")
                    # Fallback to individual deletes if batch fails
                    for item_id in ids:
                        try:
                            self.client.data_object.delete(
                                uuid=item_id,
                                class_name=collection_name
                            )
                        except Exception as inner_e:
                            print(f"Error deleting object {item_id}: {inner_e}")
            
            elif filter:
                # Delete by filter
                try:
                    where_clause = self._convert_filter_to_where(filter)
                    
                    # In 3.x, we can delete objects directly using where clause
                    self.client.batch.delete_objects(
                        class_name=collection_name,
                        where=where_clause
                    )
                except Exception as e:
                    print(f"Error in batch delete by filter: {e}")
                    
                    # Fallback: Get the IDs and delete individually
                    try:
                        # Get the IDs of objects matching the filter
                        result = (self.client.query
                                .get(collection_name, ["id"])
                                .with_where(where_clause)
                                .do())
                        
                        if result and "data" in result and "Get" in result["data"] and collection_name in result["data"]["Get"]:
                            items = result["data"]["Get"][collection_name]
                            for item in items:
                                try:
                                    self.client.data_object.delete(
                                        uuid=item["id"],
                                        class_name=collection_name
                                    )
                                except Exception as inner_e:
                                    print(f"Error deleting object {item['id']}: {inner_e}")
                    except Exception as query_e:
                        print(f"Error querying objects for deletion: {query_e}")
        
        except Exception as e:
            print(f"Error in delete: {e}")

    def reset(self):
        # Reset the database by deleting all classes
        try:
            schema = self.client.schema.get()
            for class_obj in schema["classes"]:
                class_name = class_obj["class"]
                self.client.schema.delete_class(class_name)
            return True
        except Exception as e:
            print(f"Error in reset: {e}")
            return False

    def _convert_filter_to_where(self, filter: dict) -> dict:
        """Convert Open WebUI filter format to Weaviate where clause"""
        if not filter:
            return {}
        
        # Handle simple equality filters
        if len(filter) == 1 and list(filter.keys())[0] in ["id", "chunk_id", "file_id", "file_name"]:
            key = list(filter.keys())[0]
            value = filter[key]
            
            if isinstance(value, str):
                return {
                    "path": [key],
                    "operator": "Equal",
                    "valueString": value
                }
            elif isinstance(value, int):
                return {
                    "path": [key],
                    "operator": "Equal",
                    "valueInt": value
                }
        
        # Handle complex filters (this is a simplified implementation)
        # For a production system, you'd need to map the full Open WebUI filter grammar to Weaviate's query language
        # This example only handles basic cases
        
        # Example of AND operation
        if "$and" in filter:
            operands = [self._convert_filter_to_where(subfilter) for subfilter in filter["$and"]]
            return {
                "operator": "And",
                "operands": operands
            }
        
        # Example of OR operation
        if "$or" in filter:
            operands = [self._convert_filter_to_where(subfilter) for subfilter in filter["$or"]]
            return {
                "operator": "Or",
                "operands": operands
            }
        
        # Default empty filter
        return {}
