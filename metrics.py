"""
Metrics for measuring persona drift in conversations.
Implements the metrics described in the proposal:
- Persona consistency (embedding similarity)
- Contradiction rate (NLI-based)
- Drift Index (temporal divergence)
- Conversation quality (BERTScore)
"""

import numpy as np
from typing import List, Dict, Optional
import warnings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Persona consistency will use simple keyword matching.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. NLI-based contradiction detection disabled.")

try:
    import torch
except ImportError:  # pragma: no cover - torch required for GPU support
    torch = None

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    warnings.warn("bert-score not available. BERTScore metric disabled.")


class PersonaDriftMetrics:
    """Computes persona drift metrics for conversation analysis."""
    
    def __init__(self, persona_description: str, use_gpu: bool = False):
        """
        Initialize metrics calculator.
        
        Args:
            persona_description: The persona/system prompt being evaluated
            use_gpu: Whether to use GPU for model inference
        """
        self.persona_description = persona_description
        self.use_gpu = use_gpu
        
        # Initialize models
        self.embedding_model = None
        self.nli_pipeline = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                warnings.warn(f"Could not load SentenceTransformer: {e}")
        
        if TRANSFORMERS_AVAILABLE:
            try:
                device_index = -1
                if use_gpu and torch is not None and torch.cuda.is_available():
                    device_index = 0

                self.nli_pipeline = pipeline(
                    "zero-shot-classification",
                    model="roberta-large-mnli",
                    device=device_index
                )
            except Exception as e:
                warnings.warn(f"Could not load NLI model: {e}")
    
    def persona_consistency(self, responses: List[str]) -> List[float]:
        """
        Compute persona consistency score for each response.
        
        Uses cosine similarity between response embeddings and persona embedding.
        Returns list of scores (0-1, higher is more consistent).
        """
        if not self.embedding_model:
            warnings.warn("Embedding model not available. Returning zeros.")
            return [0.0] * len(responses)
        
        try:
            # Get embeddings
            persona_emb = self.embedding_model.encode([self.persona_description])[0]
            response_embs = self.embedding_model.encode(responses)
            
            # Compute cosine similarity
            similarities = []
            for resp_emb in response_embs:
                cos_sim = np.dot(persona_emb, resp_emb) / (
                    np.linalg.norm(persona_emb) * np.linalg.norm(resp_emb)
                )
                # Normalize to 0-1 range (cosine similarity is -1 to 1)
                similarities.append((cos_sim + 1) / 2)
            
            return similarities
        except Exception as e:
            warnings.warn(f"Error computing persona consistency: {e}")
            return [0.0] * len(responses)
    
    def contradiction_rate(self, responses: List[str]) -> List[float]:
        """
        Detect contradictions between persona and responses using NLI.
        
        Returns list of contradiction scores (0 = no contradiction, 1 = contradiction).
        """
        if not self.nli_pipeline:
            warnings.warn("NLI model not available. Returning zeros.")
            return [0.0] * len(responses)
        
        contradiction_scores = []
        
        for response in responses:
            try:
                # Use zero-shot classification to check if response contradicts persona
                # Format: premise is persona, hypothesis is response
                result = self.nli_pipeline(
                    response,
                    [f"This contradicts: {self.persona_description}", 
                     f"This is consistent with: {self.persona_description}",
                     f"This is neutral regarding: {self.persona_description}"],
                )
                
                # Get contradiction probability (first label)
                contradiction_prob = result['scores'][0] if result['labels'][0].startswith('This contradicts') else 0.0
                contradiction_scores.append(contradiction_prob)
            except Exception as e:
                warnings.warn(f"Error checking contradiction for response: {e}")
                contradiction_scores.append(0.0)
        
        return contradiction_scores
    
    def drift_index(self, responses: List[str], early_turns: int = 3) -> List[float]:
        """
        Compute drift index: semantic divergence from early-turn responses.
        
        Args:
            responses: List of responses in conversation order
            early_turns: Number of early turns to use as baseline
        
        Returns:
            List of drift scores (0 = no drift, higher = more drift)
        """
        if len(responses) < early_turns:
            return [0.0] * len(responses)
        
        if not self.embedding_model:
            warnings.warn("Embedding model not available. Returning zeros.")
            return [0.0] * len(responses)
        
        try:
            # Get embeddings for all responses
            all_embs = self.embedding_model.encode(responses)
            
            # Compute average embedding of early turns
            early_embs = all_embs[:early_turns]
            baseline_emb = np.mean(early_embs, axis=0)
            
            # Compute divergence for each turn
            drift_scores = []
            for resp_emb in all_embs:
                # Cosine distance (1 - cosine similarity)
                cos_sim = np.dot(baseline_emb, resp_emb) / (
                    np.linalg.norm(baseline_emb) * np.linalg.norm(resp_emb)
                )
                # Drift is the distance from baseline
                drift = 1 - cos_sim
                drift_scores.append(drift)
            
            return drift_scores
        except Exception as e:
            warnings.warn(f"Error computing drift index: {e}")
            return [0.0] * len(responses)
    
    def conversation_quality(self, responses: List[str], references: Optional[List[str]] = None) -> List[float]:
        """
        Compute conversation quality using BERTScore.
        
        Args:
            responses: Generated responses
            references: Reference responses (if None, uses previous response as reference)
        
        Returns:
            List of BERTScore F1 scores
        """
        if not BERTSCORE_AVAILABLE:
            warnings.warn("BERTScore not available. Returning zeros.")
            return [0.0] * len(responses)
        
        if len(responses) == 0:
            return []
        
        try:
            # If no references provided, use previous response as reference
            if references is None:
                references = [responses[0]] + responses[:-1]
            
            # Compute BERTScore
            if not self.use_gpu or torch is None:
                device = "cpu"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            P, R, F1 = bert_score(
                responses,
                references,
                lang='en',
                verbose=False,
                device=device
            )
            
            return F1.tolist()
        except Exception as e:
            warnings.warn(f"Error computing BERTScore: {e}")
            return [0.0] * len(responses)
    
    def compute_all_metrics(self, responses: List[str], early_turns: int = 3) -> Dict[str, List[float]]:
        """
        Compute all metrics for a conversation.
        
        Returns dictionary with keys:
        - persona_consistency: List of consistency scores
        - contradiction_rate: List of contradiction scores
        - drift_index: List of drift scores
        - conversation_quality: List of quality scores
        """
        return {
            'persona_consistency': self.persona_consistency(responses),
            'contradiction_rate': self.contradiction_rate(responses),
            'drift_index': self.drift_index(responses, early_turns=early_turns),
            'conversation_quality': self.conversation_quality(responses),
        }

