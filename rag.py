"""RAG system using Google Cloud Firestore with Vector Search."""
import json
from typing import List, Dict, Any, Optional
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

from config import (
    GCP_PROJECT, FIRESTORE_DATABASE, FIRESTORE_COLLECTION, EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS, TOP_K_RESULTS, USE_VERTEX_AI, GCP_LOCATION
)
from database import db

# Initialize embedding model
if USE_VERTEX_AI:
    from langchain_google_vertexai import VertexAIEmbeddings
    embeddings = VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=GCP_PROJECT,
        location=GCP_LOCATION
    )
    print(f"[RAG] Using Vertex AI Embeddings ({EMBEDDING_MODEL})")
else:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from config import GOOGLE_API_KEY
    embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{EMBEDDING_MODEL}",
        google_api_key=GOOGLE_API_KEY
    )
    print(f"[RAG] Using Google AI Embeddings ({EMBEDDING_MODEL})")


class RAGSystem:
    """RAG system using Firestore Vector Search."""

    def __init__(self):
        self._db = None
        self._initialized = False

    @property
    def firestore_db(self):
        """Lazy initialization of Firestore client."""
        if self._db is None:
            self._db = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DATABASE)
            print(f"[RAG] Connected to Firestore (Project: {GCP_PROJECT})")
        return self._db

    @property
    def collection(self):
        """Get the Firestore collection."""
        return self.firestore_db.collection(FIRESTORE_COLLECTION)

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return embeddings.embed_query(text)

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return embeddings.embed_documents(texts)

    def _format_breed_document(self, breed: Dict[str, Any]) -> str:
        """Format breed data into a searchable document."""
        parts = [f"Breed: {breed.get('Breed', 'Unknown')}"]

        if breed.get('Species'):
            parts.append(f"Species: {breed['Species']}")

        if breed.get('Breeddescription'):
            parts.append(f"Description: {breed['Breeddescription']}")

        purposes = []
        if breed.get('MeatBreed'):
            purposes.append("meat production")
        if breed.get('MilkBreed'):
            purposes.append("milk/dairy production")
        if breed.get('WoolBreed'):
            purposes.append("wool/fiber production")
        if breed.get('EggBreed'):
            purposes.append("egg production")
        if breed.get('Working'):
            purposes.append("working/draft animal")

        if purposes:
            parts.append(f"Purpose: {', '.join(purposes)}")

        return " | ".join(parts)

    def _format_species_document(self, species: Dict[str, Any], colors: List[str],
                                  patterns: List[str], categories: List[str]) -> str:
        """Format species data into a searchable document."""
        parts = [f"Species: {species.get('Species', 'Unknown')}"]

        if species.get('SingularTerm'):
            parts.append(f"Singular: {species['SingularTerm']}")
        if species.get('PluralTerm'):
            parts.append(f"Plural: {species['PluralTerm']}")
        if species.get('MaleTerm'):
            parts.append(f"Male term: {species['MaleTerm']}")
        if species.get('FemaleTerm'):
            parts.append(f"Female term: {species['FemaleTerm']}")
        if species.get('BabyTerm'):
            parts.append(f"Baby term: {species['BabyTerm']}")
        if species.get('GestationPeriod'):
            parts.append(f"Gestation period: {species['GestationPeriod']} days")

        if colors:
            parts.append(f"Available colors: {', '.join(colors[:20])}")
        if patterns:
            parts.append(f"Available patterns: {', '.join(patterns[:20])}")
        if categories:
            parts.append(f"Categories: {', '.join(categories[:15])}")

        return " | ".join(parts)

    def _check_if_indexed(self) -> bool:
        """Check if data is already indexed in Firestore."""
        docs = self.collection.limit(1).get()
        return len(list(docs)) > 0

    def _get_document_count(self) -> int:
        """Get the count of documents in the collection using aggregation."""
        try:
            count_query = self.collection.count()
            result = count_query.get()
            return result[0][0].value
        except Exception as e:
            print(f"[RAG] Warning: Could not get document count: {e}")
            return -1

    def index_database(self, force_rebuild: bool = False) -> int:
        """Index all livestock data to Firestore with embeddings."""
        # Check if already indexed
        if not force_rebuild and self._check_if_indexed():
            count = self._get_document_count()
            if count >= 0:
                print(f"[RAG] Using existing Firestore index ({count} documents)")
            else:
                print("[RAG] Using existing Firestore index")
            self._initialized = True
            return max(count, 0)

        print("[RAG] Building Firestore vector index from database...")

        # Clear existing documents if rebuilding
        if force_rebuild:
            print("[RAG] Clearing existing documents...")
            deleted_count = 0
            while True:
                docs = list(self.collection.limit(100).stream())
                if not docs:
                    break
                batch = self.firestore_db.batch()
                for doc in docs:
                    batch.delete(doc.reference)
                batch.commit()
                deleted_count += len(docs)
                print(f"[RAG] Deleted {deleted_count} documents...")

        documents = []

        # Prepare breed documents
        print("[RAG] Preparing breed documents...")
        breeds = db.get_all_breeds()
        for breed in breeds:
            content = self._format_breed_document(breed)
            documents.append({
                "id": f"breed_{breed.get('BreedLookupID', '')}",
                "content": content,
                "type": "breed",
                "breed_name": breed.get('Breed', ''),
                "species": breed.get('Species', ''),
                "species_id": str(breed.get('SpeciesID', ''))
            })

        # Prepare species documents
        print("[RAG] Preparing species documents...")
        species_list = db.get_all_species()
        for species in species_list:
            species_id = species.get('SpeciesID')
            colors = db.get_colors_for_species(species_id)
            patterns = db.get_patterns_for_species(species_id)
            categories = db.get_categories_for_species(species_id)

            content = self._format_species_document(species, colors, patterns, categories)
            documents.append({
                "id": f"species_{species_id}",
                "content": content,
                "type": "species",
                "species_name": species.get('Species', ''),
                "species_id": str(species_id)
            })

        if not documents:
            print("[RAG] No documents to index")
            return 0

        # Generate embeddings in batches and store in Firestore
        print(f"[RAG] Generating embeddings for {len(documents)} documents...")

        batch_size = 20
        total_indexed = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            contents = [doc["content"] for doc in batch_docs]

            # Generate embeddings
            batch_embeddings = self._get_embeddings_batch(contents)

            # Store in Firestore with embeddings
            batch = self.firestore_db.batch()
            for doc, embedding in zip(batch_docs, batch_embeddings):
                doc_ref = self.collection.document(doc["id"])
                batch.set(doc_ref, {
                    "content": doc["content"],
                    "type": doc["type"],
                    "metadata": {k: v for k, v in doc.items() if k not in ["id", "content", "type"]},
                    "embedding": Vector(embedding)
                })

            batch.commit()
            total_indexed += len(batch_docs)
            print(f"[RAG] Indexed {total_indexed}/{len(documents)}")

        self._initialized = True
        print(f"[RAG] Index complete: {total_indexed} documents in Firestore")
        return total_indexed

    def search(self, query: str, n_results: int = TOP_K_RESULTS,
               filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using vector similarity."""
        if not self._initialized and not self._check_if_indexed():
            self.index_database()
            self._initialized = True

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Perform vector search
        collection_ref = self.collection

        # Apply type filter if specified
        if filter_type:
            collection_ref = collection_ref.where("type", "==", filter_type)

        # Vector nearest neighbor search
        vector_query = collection_ref.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.COSINE,
            limit=n_results
        )

        results = vector_query.get()

        # Format results
        formatted = []
        for doc in results:
            data = doc.to_dict()
            formatted.append({
                "content": data.get("content", ""),
                "metadata": data.get("metadata", {}),
                "type": data.get("type", "unknown"),
                "relevance_score": 1.0  # Firestore doesn't return distance in basic query
            })

        return formatted

    def get_context_for_query(self, query: str) -> str:
        """Get formatted context string for LLM."""
        results = self.search(query)

        if not results:
            return "No relevant information found in the database."

        context_parts = ["Relevant information from the livestock database:\n"]
        for i, result in enumerate(results, 1):
            context_parts.append(f"{i}. {result['content']}")
            context_parts.append(f"   (Type: {result['type']})")
            context_parts.append("")

        return "\n".join(context_parts)


# Singleton instance
rag = RAGSystem()
