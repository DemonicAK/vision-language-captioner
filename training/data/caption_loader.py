"""Caption data loading and management.

This module handles loading captions from various file formats
and organizing them for training.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from training.data.preprocessor import CaptionPreprocessor, TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class CaptionData:
    """Container for image caption data.
    
    Attributes:
        image_id: Unique identifier for the image.
        captions: List of caption strings for this image.
    """
    image_id: str
    captions: List[str] = field(default_factory=list)
    
    def add_caption(self, caption: str) -> None:
        """Add a caption to this image."""
        self.captions.append(caption)


@dataclass
class DataSplit:
    """Container for train/val/test data splits.
    
    Attributes:
        train: Dictionary of training image_id -> captions.
        val: Dictionary of validation image_id -> captions.
        test: Dictionary of test image_id -> captions.
    """
    train: Dict[str, List[str]]
    val: Dict[str, List[str]]
    test: Dict[str, List[str]]
    
    @property
    def train_keys(self) -> List[str]:
        """List of training image IDs."""
        return list(self.train.keys())
    
    @property
    def val_keys(self) -> List[str]:
        """List of validation image IDs."""
        return list(self.val.keys())
    
    @property
    def test_keys(self) -> List[str]:
        """List of test image IDs."""
        return list(self.test.keys())
    
    @property
    def all_train_captions(self) -> List[str]:
        """Flat list of all training captions."""
        return [cap for caps in self.train.values() for cap in caps]


class CaptionLoader:
    """Loader for image caption datasets.
    
    Handles loading captions from text files, preprocessing,
    and creating train/val/test splits.
    
    Attributes:
        preprocessor: Text preprocessor instance.
        caption_preprocessor: Caption-specific preprocessor with tokens.
    
    Example:
        >>> loader = CaptionLoader()
        >>> descriptions = loader.load("/path/to/captions.txt")
        >>> splits = loader.create_splits(descriptions, train_ratio=0.8)
    """
    
    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        """Initialize the caption loader.
        
        Args:
            preprocessor: Text preprocessor instance. If None, uses default.
        """
        self._preprocessor = preprocessor or TextPreprocessor()
        self._caption_preprocessor = CaptionPreprocessor()
    
    @property
    def preprocessor(self) -> TextPreprocessor:
        """Text preprocessor instance."""
        return self._preprocessor
    
    def load(self, captions_file: str | Path) -> Dict[str, List[str]]:
        """Load captions from a text file.
        
        Expected format: image_filename,caption text
        
        Args:
            captions_file: Path to captions file.
            
        Returns:
            Dictionary mapping image_id to list of captions.
            
        Raises:
            FileNotFoundError: If captions file doesn't exist.
        """
        captions_file = Path(captions_file)
        if not captions_file.exists():
            raise FileNotFoundError(f"Captions file not found: {captions_file}")
        
        descriptions: Dict[str, List[str]] = {}
        
        with open(captions_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        lines = content.split("\n")
        
        # Skip header row if present (e.g., "image,caption")
        if lines and lines[0].strip().lower() in ("image,caption", "image, caption", "filename,caption"):
            lines = lines[1:]
        
        for line in lines:
            if len(line) < 3:
                continue
            
            tokens = line.split(",", 1)
            if len(tokens) != 2:
                continue
            
            # Extract image ID (without extension)
            image_id = tokens[0].split(".")[0]
            
            # Skip if image_id looks like a header
            if image_id.lower() in ("image", "filename", "file"):
                continue
            
            caption = tokens[1].strip()
            
            # Preprocess the caption
            cleaned_caption = self._preprocessor.preprocess(caption)
            
            if image_id not in descriptions:
                descriptions[image_id] = []
            descriptions[image_id].append(cleaned_caption)
        
        logger.info(f"Loaded {len(descriptions)} images with captions")
        return descriptions
    
    def create_splits(
        self,
        descriptions: Dict[str, List[str]],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> DataSplit:
        """Split descriptions into train/val/test sets.
        
        Args:
            descriptions: Dictionary of image_id -> captions.
            train_ratio: Fraction for training set.
            val_ratio: Fraction for validation set.
            test_ratio: Fraction for test set.
            seed: Random seed for reproducibility.
            
        Returns:
            DataSplit containing the three splits with sequence tokens added.
        """
        keys = list(descriptions.keys())
        random.Random(seed).shuffle(keys)
        
        n = len(keys)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        
        train_keys = set(keys[:n_train])
        val_keys = set(keys[n_train:n_train + n_val])
        test_keys = set(keys[n_train + n_val:])
        
        train_desc: Dict[str, List[str]] = {}
        val_desc: Dict[str, List[str]] = {}
        test_desc: Dict[str, List[str]] = {}
        
        for image_id, captions in descriptions.items():
            # Add sequence tokens to each caption
            tokenized_captions = [
                self._caption_preprocessor.add_sequence_tokens(cap)
                for cap in captions
            ]
            
            if image_id in train_keys:
                train_desc[image_id] = tokenized_captions
            elif image_id in val_keys:
                val_desc[image_id] = tokenized_captions
            elif image_id in test_keys:
                test_desc[image_id] = tokenized_captions
        
        logger.info(
            f"Created splits: train={len(train_desc)}, "
            f"val={len(val_desc)}, test={len(test_desc)}"
        )
        
        return DataSplit(train=train_desc, val=val_desc, test=test_desc)
    
    def get_max_caption_length(self, descriptions: Dict[str, List[str]]) -> int:
        """Get the maximum caption length in tokens.
        
        Args:
            descriptions: Dictionary of image_id -> captions.
            
        Returns:
            Maximum number of tokens in any caption.
        """
        max_len = 0
        for captions in descriptions.values():
            for cap in captions:
                length = len(cap.split())
                max_len = max(max_len, length)
        return max_len
