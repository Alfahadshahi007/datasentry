import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


class PolicyRAGEngine:
    def __init__(self, policy_path: str = "data_rules.txt"):
        self.policy_path = Path(policy_path)
        self.collection_name = "datasentry_governance_policies"
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.qdrant = QdrantClient(":memory:")
        self.rules: list[str] = self._load_rules()
        self.use_dense = False
        self._index_rules()

    def _load_rules(self, custom_text: str | None = None) -> list[str]:
        if custom_text and custom_text.strip():
            return [line.strip() for line in custom_text.splitlines() if line.strip()]
        if self.policy_path.exists():
            raw = self.policy_path.read_text(encoding="utf-8").splitlines()
            return [line.strip() for line in raw if line.strip()]
        return [
            "[DEFAULT-01] Purge duplicate rows and standardize text casing.",
            "[DEFAULT-02] Impute missing numeric values using regression or median.",
        ]

    def _embed_texts(self, texts: list[str], task_type: str) -> list[list[float]]:
        client = genai.Client(api_key=self.api_key)
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768,
            ),
        )
        return [emb.values for emb in result.embeddings]

    def _index_rules(self) -> None:
        if not self.rules:
            return

        if self.api_key:
            try:
                vectors = self._embed_texts(self.rules, task_type="RETRIEVAL_DOCUMENT")
                if self.qdrant.collection_exists(self.collection_name):
                    self.qdrant.delete_collection(self.collection_name)

                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
                )
                points = [
                    PointStruct(id=idx, vector=vec, payload={"rule": rule})
                    for idx, (rule, vec) in enumerate(zip(self.rules, vectors))
                ]
                self.qdrant.upsert(collection_name=self.collection_name, points=points)
                self.use_dense = True
                return
            except Exception:
                self.use_dense = False

        # Fallback: Local TF-IDF sparse index
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.rules)

    def retrieve_relevant_policies(
        self, analysis: dict, custom_policy_text: str | None = None, top_k: int = 4
    ) -> list[str]:
        """Retrieves the top-k governance policies matching the dataset's schema and anomalies."""
        if custom_policy_text and custom_policy_text.strip():
            self.rules = self._load_rules(custom_policy_text)
            self._index_rules()

        if not self.rules:
            return []

        cols = " ".join(analysis.get("column_names", []))
        null_cols = " ".join(analysis.get("missing_by_col", {}).keys())
        outlier_cols = " ".join(analysis.get("outliers_by_col", {}).keys())
        dup_term = "duplicate rows primary key" if analysis.get("duplicate_rows", 0) > 0 else ""
        query = f"columns {cols} missing {null_cols} outliers {outlier_cols} {dup_term} text formatting privacy"

        if self.use_dense and self.api_key:
            try:
                query_vec = self._embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
                hits = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    limit=top_k,
                    with_payload=True,
                ).points
                return [hit.payload["rule"] for hit in hits]
            except Exception:
                pass

        # Fallback TF-IDF retrieval
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(self.rules)
        scores = cosine_similarity(vectorizer.transform([query]), matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.rules[i] for i in top_indices if scores[i] > 0.01] or self.rules[:top_k]