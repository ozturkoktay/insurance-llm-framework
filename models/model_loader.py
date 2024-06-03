"""
Model loader module for the Insurance LLM Framework.

This module provides utilities for loading and configuring various open-source LLMs.
Supported models include LLaMA-2, Mistral, Falcon, and other HuggingFace models.
"""

import os
import logging
import gc
from typing import Dict, Any, Optional, List, Union, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel, PeftConfig
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for model repositories and settings."""

    # Get HuggingFace token from environment
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

    # Define CPU-friendly models
    CPU_FRIENDLY_MODELS = {
        "tiny-llama-1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "phi-1.5": "microsoft/phi-1_5",
        "phi-2": "microsoft/phi-2",
        "mistral-7b-bnb-4bit": "TheBloke/Mistral-7B-v0.1-GGUF",
        "llama2-7b-chat-bnb-4bit": "TheBloke/Llama-2-7B-Chat-GGUF",
    }

    # Model repositories
    MODEL_REPOS = {
        "llama2-7b": "meta-llama/Llama-2-7b-hf",
        "llama2-13b": "meta-llama/Llama-2-13b-hf",
        "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
        "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
        "falcon-7b": "tiiuae/falcon-7b",
        "falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
        **CPU_FRIENDLY_MODELS  # Add CPU-friendly models to the repository
    }

    # Model details
    MODEL_DETAILS = {
        "llama2-7b": {
            "description": "LLaMA-2 7B base model",
            "parameters": "7 billion",
            "context_length": "4096 tokens",
            "suitable_for": "General text generation and completion",
            "cpu_friendly": False
        },
        "llama2-13b": {
            "description": "LLaMA-2 13B base model",
            "parameters": "13 billion",
            "context_length": "4096 tokens",
            "suitable_for": "Higher quality general text generation",
            "cpu_friendly": False
        },
        "llama2-7b-chat": {
            "description": "LLaMA-2 7B chat-tuned model",
            "parameters": "7 billion",
            "context_length": "4096 tokens",
            "suitable_for": "Conversational applications and chat",
            "cpu_friendly": False
        },
        "llama2-13b-chat": {
            "description": "LLaMA-2 13B chat-tuned model",
            "parameters": "13 billion",
            "context_length": "4096 tokens",
            "suitable_for": "Higher quality conversational applications",
            "cpu_friendly": False
        },
        "mistral-7b": {
            "description": "Mistral 7B base model",
            "parameters": "7 billion",
            "context_length": "8192 tokens",
            "suitable_for": "General text generation with longer context",
            "cpu_friendly": False
        },
        "mistral-7b-instruct": {
            "description": "Mistral 7B instruct-tuned model",
            "parameters": "7 billion",
            "context_length": "8192 tokens",
            "suitable_for": "Instruction-following tasks with longer context",
            "cpu_friendly": False
        },
        "falcon-7b": {
            "description": "Falcon 7B base model",
            "parameters": "7 billion",
            "context_length": "2048 tokens",
            "suitable_for": "Efficient general text generation",
            "cpu_friendly": False
        },
        "falcon-7b-instruct": {
            "description": "Falcon 7B instruct-tuned model",
            "parameters": "7 billion",
            "context_length": "2048 tokens",
            "suitable_for": "Instruction-following tasks",
            "cpu_friendly": False
        },
        "tiny-llama-1b": {
            "description": "Tiny LLaMA 1.1B chat model",
            "parameters": "1.1 billion",
            "context_length": "2048 tokens",
            "suitable_for": "Fast inference on CPU, good for quick testing",
            "cpu_friendly": True
        },
        "phi-1.5": {
            "description": "Microsoft Phi-1.5",
            "parameters": "1.3 billion",
            "context_length": "2048 tokens",
            "suitable_for": "CPU-friendly, lightweight inference",
            "cpu_friendly": True
        },
        "phi-2": {
            "description": "Microsoft Phi-2",
            "parameters": "2.7 billion",
            "context_length": "2048 tokens",
            "suitable_for": "CPU-friendly, efficient generation",
            "cpu_friendly": True
        },
        "mistral-7b-bnb-4bit": {
            "description": "Mistral 7B with 4-bit quantization",
            "parameters": "7 billion (4-bit)",
            "context_length": "8192 tokens",
            "suitable_for": "Faster inference on limited hardware",
            "cpu_friendly": True
        },
        "llama2-7b-chat-bnb-4bit": {
            "description": "LLaMA-2 7B chat with 4-bit quantization",
            "parameters": "7 billion (4-bit)",
            "context_length": "4096 tokens",
            "suitable_for": "Faster chat inference on limited hardware",
            "cpu_friendly": True
        }
    }

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get a list of supported model identifiers."""
        return list(cls.MODEL_REPOS.keys())

    @classmethod
    def get_cpu_friendly_models(cls) -> List[str]:
        """Get a list of models that are optimized for CPU usage."""
        return list(cls.CPU_FRIENDLY_MODELS.keys())

    @classmethod
    def get_model_details(cls) -> Dict[str, Dict[str, str]]:
        """Get details about available models."""
        return cls.MODEL_DETAILS

    @classmethod
    def get_model_repo(cls, model_id: str) -> str:
        """Get the repository for a model ID."""
        return cls.MODEL_REPOS.get(model_id, model_id)


class ModelLoader:
    """Class for loading and configuring LLM models."""

    @staticmethod
    def _get_quantization_config(quantization: Optional[str]) -> Optional[BitsAndBytesConfig]:
        """
        Get the quantization configuration.

        Args:
            quantization: Quantization level ('4bit', '8bit', or None)

        Returns:
            BitsAndBytesConfig or None
        """
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True
            )
        return None

    @staticmethod
    def _cleanup_memory() -> None:
        """Clean up memory to help with model loading."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def load_model(
        cls,
        model_id: str,
        quantization: Optional[str] = None,
        device_map: str = "auto",
        cpu_optimize: bool = False,
        **kwargs
    ) -> Tuple[Any, Any, bool]:
        """
        Load a model and tokenizer.

        Args:
            model_id: Model identifier
            quantization: Quantization level ('4bit', '8bit', or None)
            device_map: Device mapping strategy
            cpu_optimize: Whether to apply CPU optimizations
            **kwargs: Additional arguments

        Returns:
            Tuple of (model, tokenizer, is_cpu_optimized)
        """
        logger.info(
            f"Loading model {model_id} with quantization={quantization}, device_map={device_map}")

        # Clean up memory
        cls._cleanup_memory()

        # Get the repository for the model
        repo_id = ModelConfig.get_model_repo(model_id)

        # Get quantization config
        quantization_config = cls._get_quantization_config(quantization)

        # Determine if this is a CPU-friendly model
        is_cpu_optimized = model_id in ModelConfig.get_cpu_friendly_models() or cpu_optimize

        # Special handling for CPU optimization
        if cpu_optimize and device_map == "cpu":
            logger.info("Applying CPU optimizations")

            # For small models on CPU, avoid quantization for better performance
            if model_id in ["phi-1.5", "phi-2", "tiny-llama-1b"] and quantization:
                logger.info(
                    f"Disabling quantization for {model_id} on CPU for better performance")
                quantization_config = None

        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            token=ModelConfig.HF_TOKEN,
            trust_remote_code=True
        )

        # Ensure padding token is set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = "</s>"

        # Load model with appropriate configuration
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            token=ModelConfig.HF_TOKEN,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
            **kwargs
        )

        # Apply CPU optimizations if needed
        if cpu_optimize and device_map == "cpu":
            logger.info("Applying additional CPU optimizations to model")

            # Use smaller data types for CPU
            if hasattr(model, "to") and not quantization:
                model = model.to(torch.float32)  # Use float32 for CPU

        return model, tokenizer, is_cpu_optimized


class PipelineFactory:
    """Factory class for creating text generation pipelines."""

    @staticmethod
    def create_text_generation_pipeline(
        model: Any,
        tokenizer: Any,
        cpu_optimized: bool = False
    ) -> Any:
        """
        Create a text generation pipeline.

        Args:
            model: The loaded model
            tokenizer: The tokenizer
            cpu_optimized: Whether to apply CPU optimizations

        Returns:
            Text generation pipeline
        """
        # Set up pipeline parameters
        pipeline_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "return_full_text": False,
            "task": "text-generation",
        }

        # CPU optimizations for the pipeline
        if cpu_optimized:
            logger.info("Creating CPU-optimized text generation pipeline")
            pipeline_kwargs["batch_size"] = 1  # Smaller batch size for CPU

        # Create the pipeline
        text_pipeline = pipeline(**pipeline_kwargs)

        return text_pipeline


# Module-level functions that use the classes above

def get_supported_models() -> List[str]:
    """Get a list of supported model identifiers."""
    return ModelConfig.get_supported_models()


def get_cpu_friendly_models() -> List[str]:
    """Get a list of models that are optimized for CPU usage."""
    return ModelConfig.get_cpu_friendly_models()


def get_model_details() -> Dict[str, Dict[str, str]]:
    """Get details about available models."""
    return ModelConfig.get_model_details()


def load_model(
    model_id: str,
    quantization: Optional[str] = None,
    device_map: str = "auto",
    cpu_optimize: bool = False,
    **kwargs
) -> Tuple[Any, Any, bool]:
    """
    Load a model and tokenizer.

    Args:
        model_id: Model identifier
        quantization: Quantization level ('4bit', '8bit', or None)
        device_map: Device mapping strategy
        cpu_optimize: Whether to apply CPU optimizations
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, tokenizer, is_cpu_optimized)
    """
    return ModelLoader.load_model(
        model_id=model_id,
        quantization=quantization,
        device_map=device_map,
        cpu_optimize=cpu_optimize,
        **kwargs
    )


def create_text_generation_pipeline(
    model: Any,
    tokenizer: Any,
    cpu_optimized: bool = False
) -> Any:
    """
    Create a text generation pipeline.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        cpu_optimized: Whether to apply CPU optimizations

    Returns:
        Text generation pipeline
    """
    return PipelineFactory.create_text_generation_pipeline(
        model=model,
        tokenizer=tokenizer,
        cpu_optimized=cpu_optimized
    )
