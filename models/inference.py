"""
Inference module for the Insurance LLM Framework.

This module provides utilities for generating text with loaded models
based on insurance domain prompts.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from contextlib import contextmanager

import torch
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass

class ThreadingUtils:
    """Utilities for thread-based operations like timeouts."""

    @staticmethod
    def run_with_timeout(
        func: Callable,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        timeout_seconds: int = 60
    ) -> Tuple[Any, Optional[Exception]]:
        """
        Run a function with timeout using a separate thread.

        Args:
            func: The function to run
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            timeout_seconds: Maximum seconds to wait

        Returns:
            Tuple of (result, exception)
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        result = [None]
        error = [None]
        completed = [False]

        def worker():
            try:
                result[0] = func(*args, **kwargs)
                completed[0] = True
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)

        if not completed[0]:
            if thread.is_alive():
                return None, TimeoutException(f"Function timed out after {timeout_seconds} seconds")

        if error[0] is not None:
            return None, error[0]

        return result[0], None

class SystemInfo:
    """Class for getting system information."""

    @staticmethod
    def get_cpu_count() -> int:
        """Get the number of CPU cores."""
        import multiprocessing
        return multiprocessing.cpu_count()

    @staticmethod
    def is_limited_cpu() -> bool:
        """Check if the system has limited CPU resources."""
        return SystemInfo.get_cpu_count() < 4

class GenerationParameters:
    """Class for managing text generation parameters."""

    @staticmethod
    def get_default_parameters(cpu_optimized: bool = False) -> Dict[str, Any]:
        """
        Get default generation parameters.

        Args:
            cpu_optimized: Whether to optimize for CPU

        Returns:
            Dictionary of default parameters
        """
        params = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "num_return_sequences": 1,
        }

        if cpu_optimized:

            params["max_new_tokens"] = 256

            if SystemInfo.is_limited_cpu():
                params["max_new_tokens"] = 128

        return params

    @staticmethod
    def apply_stop_sequences(
        text: str,
        stop_sequences: List[str]
    ) -> str:
        """
        Truncate text at the first occurrence of any stop sequence.

        Args:
            text: The generated text
            stop_sequences: List of sequences that should stop generation

        Returns:
            Truncated text
        """
        if not stop_sequences:
            return text

        positions = []
        for seq in stop_sequences:
            pos = text.find(seq)
            if pos != -1:
                positions.append(pos)

        if positions:
            return text[:min(positions)]

        return text

class ModelInference:
    """
    Model inference for the Insurance LLM Framework.

    This class handles text generation using loaded models with support for
    prompt templates and parameter control.
    """

    def __init__(self, model: Any, tokenizer: Any, cpu_optimized: bool = False):
        """
        Initialize the inference engine.

        Args:
            model: The loaded LLM
            tokenizer: The associated tokenizer
            cpu_optimized: Whether to use CPU optimizations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = None
        self.cpu_optimized = cpu_optimized
        self.initialize_pipeline()

        self.default_batch_size = 1 if cpu_optimized else 4

        self.slow_cpu = SystemInfo.is_limited_cpu()

        if self.slow_cpu and cpu_optimized:
            logger.info(
                "Running on a limited CPU - extra optimizations applied")

    def initialize_pipeline(self):
        """Initialize the text generation pipeline."""
        from models.model_loader import create_text_generation_pipeline

        self.pipeline = create_text_generation_pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            cpu_optimized=self.cpu_optimized
        )

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None,
        timeout_seconds: int = 120,
    ) -> List[str]:
        """
        Generate text based on a prompt.

        Args:
            prompt: The input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            do_sample: Whether to use sampling
            stop_sequences: List of sequences that will stop generation
            timeout_seconds: Maximum number of seconds to wait for generation

        Returns:
            List of generated texts
        """
        logger.info(f"Generating text with prompt length: {len(prompt)}")

        if self.cpu_optimized and self.slow_cpu and max_length > 256:
            logger.info(
                f"Reducing max_length from {max_length} to 256 for CPU performance")
            max_length = 256

        gen_kwargs = {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if not do_sample:
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 0

        try:

            def generate_text():
                return self.pipeline(
                    prompt,
                    **gen_kwargs
                )

            result, error = ThreadingUtils.run_with_timeout(
                generate_text,
                timeout_seconds=timeout_seconds
            )

            if error:
                if isinstance(error, TimeoutException):
                    logger.warning(
                        f"Generation timed out after {timeout_seconds} seconds")
                    raise TimeoutException(
                        f"Generation timed out after {timeout_seconds} seconds")
                else:
                    logger.error(f"Error during generation: {str(error)}")
                    raise error

            generated_texts = []
            for output in result:
                text = output["generated_text"]

                if stop_sequences:
                    text = GenerationParameters.apply_stop_sequences(
                        text, stop_sequences)

                generated_texts.append(text)

            return generated_texts

        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            raise

    def generate_with_streaming(
        self,
        prompt: str,
        callback: Callable[[str], None],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None,
        timeout_seconds: int = 120,
    ) -> str:
        """
        Generate text with streaming callback.

        Args:
            prompt: The input prompt
            callback: Function to call with each token
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            stop_sequences: List of sequences that will stop generation
            timeout_seconds: Maximum seconds to wait

        Returns:
            The complete generated text
        """
        logger.info(f"Streaming generation with prompt length: {len(prompt)}")

        if self.cpu_optimized and self.slow_cpu and max_length > 256:
            logger.info(
                f"Reducing max_length from {max_length} to 256 for CPU performance")
            max_length = 256

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        if hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_length,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if not do_sample:
            temperature = 1.0
            top_p = 1.0
            top_k = 0

        start_time = time.time()

        full_text = ""

        stopped = False

        try:

            generated_ids = []

            for _ in range(max_length):

                if time.time() - start_time > timeout_seconds:
                    logger.warning(
                        f"Streaming generation timed out after {timeout_seconds} seconds")
                    break

                if stopped:
                    break

                with torch.no_grad():

                    outputs = self.model(
                        input_ids if not generated_ids else torch.cat(
                            [input_ids, torch.tensor([generated_ids], device=input_ids.device)], dim=-1)
                    )
                    next_token_logits = outputs.logits[:, -1, :]

                    if do_sample:

                        next_token_logits = next_token_logits / temperature

                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(
                                next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(
                                torch.softmax(sorted_logits, dim=-1), dim=-1)

                            sorted_indices_to_remove = cumulative_probs > top_p

                            sorted_indices_to_remove[...,
                                                     1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0

                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            next_token_logits[:,
                                              indices_to_remove] = -float('Inf')

                        if top_k > 0:
                            top_k = min(top_k, next_token_logits.size(-1))
                            indices_to_remove = next_token_logits < torch.topk(
                                next_token_logits, top_k)[0][..., -1, None]
                            next_token_logits[indices_to_remove] = - \
                                float('Inf')

                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(
                            probs, num_samples=1).squeeze(1)
                    else:

                        next_token = torch.argmax(next_token_logits, dim=-1)

                next_token_text = self.tokenizer.decode(next_token)

                full_text += next_token_text

                callback(next_token_text)

                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in full_text:

                            stop_pos = full_text.find(stop_seq)
                            full_text = full_text[:stop_pos]
                            stopped = True
                            break

                generated_ids.append(next_token.item())

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            return full_text

        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            raise

    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        batch_size: Optional[int] = None,
        timeout_seconds: int = 300,
    ) -> List[str]:
        """
        Generate text for multiple prompts in batches.

        Args:
            prompts: List of input prompts
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            batch_size: Batch size for generation
            timeout_seconds: Maximum seconds to wait

        Returns:
            List of generated texts
        """
        logger.info(f"Batch generating for {len(prompts)} prompts")

        if batch_size is None:
            batch_size = self.default_batch_size

        if self.cpu_optimized:
            batch_size = 1

        if self.cpu_optimized and self.slow_cpu and max_length > 256:
            logger.info(
                f"Reducing max_length from {max_length} to 256 for CPU performance")
            max_length = 256

        gen_kwargs = {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]

            try:

                def generate_batch():
                    return self.pipeline(
                        batch_prompts,
                        **gen_kwargs
                    )

                batch_results, error = ThreadingUtils.run_with_timeout(
                    generate_batch,
                    timeout_seconds=timeout_seconds
                )

                if error:
                    if isinstance(error, TimeoutException):
                        logger.warning(
                            f"Batch generation timed out after {timeout_seconds} seconds")
                        raise TimeoutException(
                            f"Batch generation timed out after {timeout_seconds} seconds")
                    else:
                        logger.error(
                            f"Error during batch generation: {str(error)}")
                        raise error

                for output in batch_results:
                    results.append(output["generated_text"])

            except Exception as e:
                logger.error(f"Error in batch_generate: {str(e)}")

                results.extend([""] * len(batch_prompts))

        return results
