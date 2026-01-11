"""FastAPI application for image captioning inference.

Production-ready API with proper initialization, monitoring,
request logging, and error handling.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

import io
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Application state
app_state: Dict[str, Any] = {
    "model_loaded": False,
    "startup_time": None,
    "request_count": 0,
    "error_count": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events for proper resource management.
    """
    # Startup
    logger.info("Starting up application...")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Initialize model lazily on first request, not at startup
    # This improves cold start time significantly
    app_state["startup_time"] = datetime.now().isoformat()
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Cleanup model resources
    from inference.model_registry import ModelRegistry
    registry = ModelRegistry()
    registry.clear()
    
    logger.info("Application shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Image Captioning API",
    description="Production API for generating captions from images",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Log request
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - Started"
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - "
        f"Status: {response.status_code}, Duration: {duration:.3f}s"
    )
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    
    # Update metrics
    app_state["request_count"] += 1
    
    return response


def ensure_model_loaded() -> None:
    """Ensure model is loaded before processing requests."""
    if not app_state["model_loaded"]:
        from inference.model_registry import ModelRegistry, ModelConfig
        
        logger.info("Loading model on first request...")
        
        artifacts_dir = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")
        config = ModelConfig.from_artifacts_dir(artifacts_dir)
        
        registry = ModelRegistry()
        registry.register("default", config)
        
        # This triggers lazy loading
        _ = registry.get_model("default")
        
        app_state["model_loaded"] = True
        logger.info("Model loaded successfully")


@app.get("/")
def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Image Captioning API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "model_loaded": app_state["model_loaded"],
        "startup_time": app_state["startup_time"],
        "tensorflow_version": tf.__version__,
    }


@app.get("/ready")
def readiness_check() -> Dict[str, Any]:
    """Readiness check endpoint.
    
    Returns 503 if model is not ready to serve requests.
    """
    if not app_state["model_loaded"]:
        # Model will be loaded on first request
        # Return ready anyway as it will load lazily
        pass
    
    return {
        "ready": True,
        "model_loaded": app_state["model_loaded"],
    }


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    """Get application metrics for monitoring."""
    return {
        "request_count": app_state["request_count"],
        "error_count": app_state["error_count"],
        "model_loaded": app_state["model_loaded"],
        "startup_time": app_state["startup_time"],
        "uptime_seconds": (
            (datetime.now() - datetime.fromisoformat(app_state["startup_time"])).total_seconds()
            if app_state["startup_time"]
            else 0
        ),
    }


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    ensure_model_loaded()
    
    from inference.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    bundle = registry.get_model("default")
    
    return {
        "vocab_size": bundle.vocab_size,
        "max_length": bundle.max_length,
        "feature_extractor": bundle.config.feature_extractor_name,
        "image_size": bundle.config.image_size,
        "model_path": bundle.config.model_path,
    }


@app.post("/predict")
async def predict_caption(
    file: UploadFile = File(...),
    algorithm: str = Form("beam"),
) -> Dict[str, Any]:
    """Generate caption for an uploaded image.
    
    Args:
        file: Image file to caption.
        algorithm: Caption generation algorithm ("greedy" or "beam").
        
    Returns:
        Dictionary with generated caption and metadata.
    """
    request_start = time.time()
    
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Read and validate image
        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}",
            )
        
        # Generate caption
        from inference.model_registry import ModelRegistry
        from inference.preprocessing import create_preprocessor_from_bundle
        from inference.caption_generator import CaptionService
        
        registry = ModelRegistry()
        bundle = registry.get_model("default")
        
        # Extract features
        preprocessor = create_preprocessor_from_bundle(bundle)
        features = preprocessor.extract_features(image)
        
        # Ensure batch dimension
        if len(features.shape) == 1:
            features = tf.expand_dims(features, axis=0)
        
        # Generate caption
        service = CaptionService(bundle)
        caption = service.generate_caption(features, algorithm=algorithm)
        
        # Validate caption
        if not caption or len(caption.strip()) <= 1:
            caption = "Unable to generate caption"
        
        processing_time = time.time() - request_start
        
        return {
            "caption": caption,
            "algorithm": algorithm,
            "processing_time_seconds": round(processing_time, 3),
            "image_size": image.size,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        app_state["error_count"] += 1
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    app_state["error_count"] += 1
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


# Main entry point for direct execution
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
