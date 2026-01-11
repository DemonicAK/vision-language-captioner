"""Model versioning and artifact management.

This module provides model versioning capabilities for tracking
different model versions and their metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a model version.
    
    Attributes:
        version: Semantic version string (e.g., "1.0.0").
        model_hash: SHA256 hash of the model file.
        created_at: Timestamp of version creation.
        run_id: Associated training run ID.
        metrics: Training/evaluation metrics.
        config: Model configuration snapshot.
        description: Human-readable description.
        tags: Tags for categorization.
    """
    version: str
    model_hash: str
    created_at: str
    run_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(**data)


class ModelVersionManager:
    """Manager for model versions.
    
    Tracks model versions, their metadata, and provides
    version management operations.
    
    Example:
        >>> manager = ModelVersionManager("models/")
        >>> manager.register_version(
        ...     version="1.0.0",
        ...     model_path="artifacts/model.keras",
        ...     metrics={"bleu4": 0.25},
        ... )
        >>> latest = manager.get_latest_version()
    """
    
    REGISTRY_FILE = "model_registry.json"
    
    def __init__(self, versions_dir: str | Path) -> None:
        """Initialize version manager.
        
        Args:
            versions_dir: Directory for storing model versions.
        """
        self._versions_dir = Path(versions_dir)
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._versions_dir / self.REGISTRY_FILE
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, ModelVersion]:
        """Load version registry from disk."""
        if not self._registry_path.exists():
            return {}
        
        with open(self._registry_path, "r") as f:
            data = json.load(f)
        
        return {
            version: ModelVersion.from_dict(info)
            for version, info in data.items()
        }
    
    def _save_registry(self) -> None:
        """Save version registry to disk."""
        data = {
            version: info.to_dict()
            for version, info in self._registry.items()
        }
        
        with open(self._registry_path, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def register_version(
        self,
        version: str,
        model_path: str | Path,
        vocab_dir: Optional[str | Path] = None,
        run_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        copy_artifacts: bool = True,
    ) -> ModelVersion:
        """Register a new model version.
        
        Args:
            version: Version string (e.g., "1.0.0").
            model_path: Path to model file.
            vocab_dir: Path to vocabulary directory.
            run_id: Associated training run ID.
            metrics: Model metrics.
            config: Model configuration.
            description: Version description.
            tags: Version tags.
            copy_artifacts: Whether to copy artifacts to versions directory.
            
        Returns:
            Created ModelVersion.
            
        Raises:
            ValueError: If version already exists.
        """
        if version in self._registry:
            raise ValueError(f"Version {version} already exists")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create version directory
        version_dir = self._versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy artifacts if requested
        if copy_artifacts:
            dest_model = version_dir / model_path.name
            shutil.copy2(model_path, dest_model)
            
            if vocab_dir:
                vocab_dir = Path(vocab_dir)
                dest_vocab = version_dir / "vocab"
                if dest_vocab.exists():
                    shutil.rmtree(dest_vocab)
                shutil.copytree(vocab_dir, dest_vocab)
        
        # Compute model hash
        model_hash = self._compute_file_hash(model_path)
        
        # Create version metadata
        version_info = ModelVersion(
            version=version,
            model_hash=model_hash,
            created_at=datetime.now().isoformat(),
            run_id=run_id,
            metrics=metrics or {},
            config=config or {},
            description=description,
            tags=tags or [],
        )
        
        # Save version metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(version_info.to_dict(), f, indent=2)
        
        # Update registry
        self._registry[version] = version_info
        self._save_registry()
        
        logger.info(f"Registered model version: {version}")
        return version_info
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get version metadata.
        
        Args:
            version: Version string.
            
        Returns:
            ModelVersion or None if not found.
        """
        return self._registry.get(version)
    
    def get_latest_version(self) -> Optional[ModelVersion]:
        """Get the latest registered version.
        
        Returns:
            Latest ModelVersion or None.
        """
        if not self._registry:
            return None
        
        # Sort by creation time
        versions = sorted(
            self._registry.values(),
            key=lambda v: v.created_at,
            reverse=True,
        )
        return versions[0]
    
    def list_versions(self) -> List[ModelVersion]:
        """List all registered versions.
        
        Returns:
            List of ModelVersion objects.
        """
        return list(self._registry.values())
    
    def get_version_path(self, version: str) -> Optional[Path]:
        """Get path to version directory.
        
        Args:
            version: Version string.
            
        Returns:
            Path to version directory or None.
        """
        if version not in self._registry:
            return None
        return self._versions_dir / version
    
    def delete_version(self, version: str) -> None:
        """Delete a model version.
        
        Args:
            version: Version to delete.
        """
        if version not in self._registry:
            raise KeyError(f"Version not found: {version}")
        
        # Remove directory
        version_dir = self._versions_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Remove from registry
        del self._registry[version]
        self._save_registry()
        
        logger.info(f"Deleted model version: {version}")
    
    def compare_versions(
        self,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """Compare two model versions.
        
        Args:
            version1: First version.
            version2: Second version.
            
        Returns:
            Comparison results.
        """
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)
        
        if v1 is None or v2 is None:
            raise KeyError("One or both versions not found")
        
        # Compare metrics
        metrics_diff = {}
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for metric in all_metrics:
            val1 = v1.metrics.get(metric)
            val2 = v2.metrics.get(metric)
            if val1 is not None and val2 is not None:
                metrics_diff[metric] = {
                    version1: val1,
                    version2: val2,
                    "diff": val2 - val1,
                    "improvement": val2 > val1,
                }
        
        return {
            "versions": [version1, version2],
            "metrics_comparison": metrics_diff,
            "same_model": v1.model_hash == v2.model_hash,
            "time_diff_seconds": (
                datetime.fromisoformat(v2.created_at)
                - datetime.fromisoformat(v1.created_at)
            ).total_seconds(),
        }


def create_deployment_manifest(
    version_manager: ModelVersionManager,
    version: str,
    deployment_name: str = "image-captioning",
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Create deployment manifest for a model version.
    
    Args:
        version_manager: Version manager instance.
        version: Version to deploy.
        deployment_name: Name for the deployment.
        output_path: Path to save manifest (optional).
        
    Returns:
        Deployment manifest dictionary.
    """
    version_info = version_manager.get_version(version)
    if version_info is None:
        raise KeyError(f"Version not found: {version}")
    
    version_path = version_manager.get_version_path(version)
    
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": deployment_name,
            "labels": {
                "app": deployment_name,
                "model-version": version,
            },
            "annotations": {
                "model-hash": version_info.model_hash,
                "created-at": version_info.created_at,
                "run-id": version_info.run_id or "unknown",
            },
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": deployment_name,
                },
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": deployment_name,
                        "model-version": version,
                    },
                },
                "spec": {
                    "containers": [
                        {
                            "name": deployment_name,
                            "image": f"{deployment_name}:{version}",
                            "env": [
                                {
                                    "name": "MODEL_VERSION",
                                    "value": version,
                                },
                            ],
                            "ports": [
                                {"containerPort": 8000},
                            ],
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000,
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10,
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000,
                                },
                                "initialDelaySeconds": 15,
                                "periodSeconds": 20,
                            },
                        },
                    ],
                },
            },
        },
    }
    
    if output_path:
        import yaml
        
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False)
        
        logger.info(f"Saved deployment manifest to {output_path}")
    
    return manifest
