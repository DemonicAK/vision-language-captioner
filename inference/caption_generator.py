"""Caption generation for inference.

This module provides caption generation algorithms that work
with the model registry and avoid global state.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

if TYPE_CHECKING:
    from inference.model_registry import ModelBundle

logger = logging.getLogger(__name__)


class BaseCaptionGenerator(ABC):
    """Abstract base class for caption generators.
    
    Defines interface for caption generation algorithms.
    """
    
    def __init__(
        self,
        model: Any,
        word_to_index: Dict[str, int],
        index_to_word: Dict[int, str],
        max_length: int,
    ) -> None:
        """Initialize generator.
        
        Args:
            model: Caption model.
            word_to_index: Vocabulary mapping.
            index_to_word: Reverse vocabulary mapping.
            max_length: Maximum sequence length.
        """
        self._model = model
        self._word_to_index = word_to_index
        self._index_to_word = index_to_word
        self._max_length = max_length
    
    @abstractmethod
    def generate(self, image_features: np.ndarray) -> List[str]:
        """Generate caption from image features.
        
        Args:
            image_features: Image feature vector.
            
        Returns:
            List of words in the caption.
        """
        pass
    
    def generate_caption(self, image_features: np.ndarray) -> str:
        """Generate caption as string.
        
        Args:
            image_features: Image feature vector.
            
        Returns:
            Caption string.
        """
        words = self.generate(image_features)
        return " ".join(words)


class GreedyGenerator(BaseCaptionGenerator):
    """Greedy caption generation.
    
    Selects the most probable word at each step.
    """
    
    def __init__(
        self,
        model: Any,
        word_to_index: Dict[str, int],
        index_to_word: Dict[int, str],
        max_length: int,
        repetition_penalty: bool = True,
    ) -> None:
        """Initialize greedy generator.
        
        Args:
            model: Caption model.
            word_to_index: Vocabulary mapping.
            index_to_word: Reverse vocabulary mapping.
            max_length: Maximum sequence length.
            repetition_penalty: Whether to penalize repetitions.
        """
        super().__init__(model, word_to_index, index_to_word, max_length)
        self._repetition_penalty = repetition_penalty
    
    def generate(self, image_features: np.ndarray) -> List[str]:
        """Generate caption using greedy search.
        
        Args:
            image_features: Image feature vector.
            
        Returns:
            List of caption words.
        """
        in_text = ["startseq"]
        
        for _ in range(self._max_length + 1):
            # Encode current sequence
            seq = [
                self._word_to_index[w]
                for w in in_text
                if w in self._word_to_index
            ]
            seq = pad_sequences([seq], maxlen=self._max_length, padding="post")
            
            # Predict next word
            preds = self._model.predict(
                [image_features, seq],
                verbose=0,
            )[0]
            
            next_id = int(np.argmax(preds))
            next_word = self._index_to_word.get(next_id)
            
            # Stop conditions
            if next_word is None:
                break
            if next_word == "endseq":
                break
            if self._repetition_penalty and next_word in in_text[-2:]:
                break
            
            in_text.append(next_word)
        
        # Remove startseq
        return in_text[1:]


class BeamSearchGenerator(BaseCaptionGenerator):
    """Beam search caption generation.
    
    Maintains multiple hypotheses and selects the best.
    """
    
    def __init__(
        self,
        model: Any,
        word_to_index: Dict[str, int],
        index_to_word: Dict[int, str],
        max_length: int,
        beam_width: int = 3,
        length_penalty: float = 0.7,
    ) -> None:
        """Initialize beam search generator.
        
        Args:
            model: Caption model.
            word_to_index: Vocabulary mapping.
            index_to_word: Reverse vocabulary mapping.
            max_length: Maximum sequence length.
            beam_width: Number of beams to maintain.
            length_penalty: Penalty for length normalization.
        """
        super().__init__(model, word_to_index, index_to_word, max_length)
        self._beam_width = beam_width
        self._length_penalty = length_penalty
    
    def generate(self, image_features: np.ndarray) -> List[str]:
        """Generate caption using beam search.
        
        Args:
            image_features: Image feature vector.
            
        Returns:
            List of caption words.
        """
        start_seq = ["startseq"]
        sequences = [(start_seq, 0.0)]
        
        for _ in range(self._max_length):
            all_candidates = []
            
            for seq, score in sequences:
                # Encode sequence
                seq_ids = [
                    self._word_to_index[w]
                    for w in seq
                    if w in self._word_to_index
                ]
                seq_ids = pad_sequences(
                    [seq_ids], maxlen=self._max_length, padding="post"
                )
                
                # Predict
                preds = self._model.predict(
                    [image_features, seq_ids],
                    verbose=0,
                )[0]
                
                # Get top candidates
                top_indices = np.argsort(preds)[-self._beam_width:]
                
                for next_id in top_indices:
                    next_word = self._index_to_word.get(next_id)
                    if next_word is None:
                        continue
                    
                    new_score = score - np.log(preds[next_id] + 1e-10)
                    
                    if next_word == "endseq":
                        all_candidates.append((seq + [next_word], new_score))
                        continue
                    
                    # Repetition guard
                    if next_word in seq[-2:]:
                        continue
                    
                    all_candidates.append((seq + [next_word], new_score))
            
            if not all_candidates:
                break
            
            # Keep top beams
            sequences = sorted(all_candidates, key=lambda x: x[1])[:self._beam_width]
            
            # Stop if all ended
            if all(seq[-1] == "endseq" for seq, _ in sequences):
                break
        
        # Get best sequence
        best_seq = sequences[0][0]
        return [w for w in best_seq if w not in ["startseq", "endseq"]]


class CaptionService:
    """High-level caption generation service.
    
    Provides a clean interface for caption generation with
    support for multiple algorithms and model bundles.
    
    Example:
        >>> service = CaptionService(model_bundle)
        >>> caption = service.generate_caption(image, algorithm="beam")
    """
    
    def __init__(self, bundle: "ModelBundle") -> None:
        """Initialize caption service.
        
        Args:
            bundle: Model bundle containing model and vocabulary.
        """
        self._bundle = bundle
        self._generators = self._create_generators()
    
    def _create_generators(self) -> Dict[str, BaseCaptionGenerator]:
        """Create caption generators."""
        return {
            "greedy": GreedyGenerator(
                model=self._bundle.model,
                word_to_index=self._bundle.word_to_index,
                index_to_word=self._bundle.index_to_word,
                max_length=self._bundle.max_length,
            ),
            "beam": BeamSearchGenerator(
                model=self._bundle.model,
                word_to_index=self._bundle.word_to_index,
                index_to_word=self._bundle.index_to_word,
                max_length=self._bundle.max_length,
            ),
        }
    
    def generate_caption(
        self,
        image_features: np.ndarray,
        algorithm: str = "beam",
    ) -> str:
        """Generate caption from image features.
        
        Args:
            image_features: Image feature vector.
            algorithm: Generation algorithm ("greedy" or "beam").
            
        Returns:
            Generated caption string.
            
        Raises:
            ValueError: If algorithm is unknown.
        """
        algorithm = algorithm.lower()
        if algorithm not in self._generators:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Available: {list(self._generators.keys())}"
            )
        
        generator = self._generators[algorithm]
        return generator.generate_caption(image_features)
    
    @property
    def available_algorithms(self) -> List[str]:
        """Get list of available algorithms."""
        return list(self._generators.keys())


# Legacy compatibility functions
def GreedySearch(image_features: np.ndarray) -> List[str]:
    """Legacy greedy search function.
    
    Uses lazy loading to avoid import-time model loading.
    """
    from inference.model_registry import _LazyModelLoader
    
    generator = GreedyGenerator(
        model=_LazyModelLoader.get_model(),
        word_to_index=_LazyModelLoader.get_word_to_index(),
        index_to_word=_LazyModelLoader.get_index_to_word(),
        max_length=_LazyModelLoader.get_max_length(),
    )
    return generator.generate(image_features)


def BeamSearch(image_features: np.ndarray, beam_width: int = 3) -> List[str]:
    """Legacy beam search function.
    
    Uses lazy loading to avoid import-time model loading.
    """
    from inference.model_registry import _LazyModelLoader
    
    generator = BeamSearchGenerator(
        model=_LazyModelLoader.get_model(),
        word_to_index=_LazyModelLoader.get_word_to_index(),
        index_to_word=_LazyModelLoader.get_index_to_word(),
        max_length=_LazyModelLoader.get_max_length(),
        beam_width=beam_width,
    )
    return generator.generate(image_features)


def run_caption_algo(algo_name: str, image_features: np.ndarray) -> List[str]:
    """Legacy algorithm dispatcher."""
    algo_name = algo_name.lower()
    
    if algo_name == "greedy":
        return GreedySearch(image_features)
    elif algo_name == "beam":
        return BeamSearch(image_features)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def prediction_pipeline(
    image: Any,
    algo_name: str = "beam",
) -> str:
    """Legacy prediction pipeline.
    
    Args:
        image: PIL Image object.
        algo_name: Algorithm name ("greedy" or "beam").
        
    Returns:
        Generated caption string.
    """
    try:
        from inference.preprocessing import image_preprocessor
        import tensorflow as tf
        
        image_features = image_preprocessor(image)
        
        # Ensure correct shape
        if len(image_features.shape) == 1:
            image_features = tf.expand_dims(image_features, axis=0)
        
        seq = run_caption_algo(algo_name, image_features)
        caption = " ".join(seq)
        
        if caption is None or len(caption.strip()) <= 1:
            return "caption not generated by model"
        
        return caption
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "no caption"
