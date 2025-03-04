"""
Insurance LLM Framework

A web application for prompt engineering and evaluation of LLMs for insurance tasks.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import framework components
from models.model_loader import (
    get_supported_models,
    get_model_details,
    load_model,
    get_cpu_friendly_models
)
from models.inference import ModelInference
from prompts.library import get_prompt_library, PromptTemplate
from prompts.strategies import create_prompt_strategy
from evaluation.metrics import get_metrics_manager
from evaluation.human_eval import get_human_evaluation_manager
from evaluation.benchmarks import (
    get_benchmark_manager,
    create_policy_summary_benchmark,
    create_claim_response_benchmark
)
from utils.dataframe_utils import prepare_dataframe_for_display

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TorchUtils:
    """Utility class for torch-related operations."""

    @staticmethod
    def get_torch():
        """Lazily import torch to avoid conflicts with Streamlit's file watcher."""
        import torch
        return torch

    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is available."""
        torch = TorchUtils.get_torch()
        return torch.cuda.is_available()

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get GPU information if available."""
        torch = TorchUtils.get_torch()
        if not torch.cuda.is_available():
            return {}

        info = {
            "name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda
        }

        # Free memory in GB
        free_memory = torch.cuda.get_device_properties(
            0).total_memory - torch.cuda.memory_allocated(0)
        info["free_memory_gb"] = free_memory / (1024**3)

        return info

    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory if CUDA is available."""
        if TorchUtils.is_cuda_available():
            torch = TorchUtils.get_torch()
            torch.cuda.empty_cache()


class SessionState:
    """Class to manage Streamlit session state initialization and access."""

    @staticmethod
    def initialize():
        """Initialize session state variables."""
        if "model" not in st.session_state:
            st.session_state.model = None

        if "tokenizer" not in st.session_state:
            st.session_state.tokenizer = None

        if "inference_engine" not in st.session_state:
            st.session_state.inference_engine = None

        if "current_tab" not in st.session_state:
            st.session_state.current_tab = "model_selection"

        if "generated_outputs" not in st.session_state:
            st.session_state.generated_outputs = []

        if "evaluation_results" not in st.session_state:
            st.session_state.evaluation_results = {}

    @staticmethod
    def set_tab(tab_name: str):
        """Set the current tab."""
        st.session_state.current_tab = tab_name


class DataLoader:
    """Class to handle loading sample data."""

    @staticmethod
    def load_sample_data(data_type: str) -> str:
        """
        Load sample data for a specified type.

        Args:
            data_type: Type of data to load (e.g., 'policy', 'claim')

        Returns:
            Sample data text
        """
        data_dir = Path("data")

        if data_type == "policy":
            file_path = data_dir / "policies" / "sample_auto_policy.txt"
        elif data_type == "claim":
            file_path = data_dir / "claims" / "sample_auto_claim.txt"
        elif data_type == "customer_inquiry":
            file_path = data_dir / "communications" / "sample_customer_inquiry.txt"
        else:
            return "Sample data not found."

        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading sample data {file_path}: {str(e)}")
            return f"Error loading sample data: {str(e)}"


class ModelSelectionPage:
    """Class to handle the model selection page UI and logic."""

    def __init__(self):
        self.torch_utils = TorchUtils()

    def _display_system_info(self):
        """Display system information."""
        st.subheader("System Information")

        # Get torch
        torch = self.torch_utils.get_torch()

        # Get more CPU info
        import platform
        import multiprocessing

        system_info = {
            "Python Version": os.sys.version.split()[0],
            "Streamlit Version": st.__version__,
            "PyTorch Version": torch.__version__,
            "CPU Cores": multiprocessing.cpu_count(),
            "Operating System": platform.system() + " " + platform.release(),
            "CUDA Available": "Yes" if torch.cuda.is_available() else "No",
        }

        if torch.cuda.is_available():
            gpu_info = self.torch_utils.get_gpu_info()
            system_info["GPU"] = gpu_info["name"]
            system_info["CUDA Version"] = gpu_info["cuda_version"]
            system_info["GPU Free Memory"] = f"{gpu_info['free_memory_gb']:.2f} GB"
        else:
            system_info["RAM"] = f"{os.popen('free -h').readlines()[1].split()[1]}"

        # Display as a table
        system_df = pd.DataFrame(
            list(system_info.items()),
            columns=["Property", "Value"]
        )
        system_df = prepare_dataframe_for_display(system_df)
        st.table(system_df)

    def _display_model_selection_ui(self):
        """Display the model selection UI."""
        st.subheader("Model Selection")

        # Get available models
        available_models = get_supported_models()
        model_details = get_model_details()
        cpu_friendly_models = get_cpu_friendly_models()

        # Check if CUDA is available
        has_cuda = self.torch_utils.is_cuda_available()

        if not has_cuda:
            st.warning(
                "⚠️ Running on CPU only. Generation will be SLOW with large models. Smaller models like Phi-2 or TinyLLaMA are recommended."
            )

            # Show CPU-friendly models section
            st.info("💡 CPU-Friendly Models Recommended")

            # Prioritize CPU-friendly models
            model_id = st.selectbox(
                "Choose a model",
                available_models,
                index=available_models.index(
                    "phi-2") if "phi-2" in available_models else 0,
                help="Select a model - smaller models (1-3B parameters) work best on CPU"
            )

            # Show CPU warning for large models
            if model_id not in cpu_friendly_models and "13b" in model_id:
                st.error(
                    "⚠️ This is a very large model (13B parameters) and will be EXTREMELY slow on CPU. It may appear to get stuck.")
            elif model_id not in cpu_friendly_models and "7b" in model_id:
                st.warning(
                    "⚠️ This is a large model (7B parameters) and will be very slow on CPU. Consider using a smaller model.")
        else:
            # GPU is available
            st.info("GPU detected: Using GPU for model inference")

            # Get GPU info
            gpu_info = self.torch_utils.get_gpu_info()
            free_memory_gb = gpu_info["free_memory_gb"]

            st.info(
                f"GPU: {gpu_info['name']} with {free_memory_gb:.2f} GB free memory")

            if free_memory_gb < 8:
                st.warning(
                    f"⚠️ Only {free_memory_gb:.2f}GB of GPU memory available. Consider using a smaller model or increasing quantization to 8-bit.")

            # Standard model selection for GPU
            model_id = st.selectbox(
                "Choose a model",
                available_models,
                index=2 if "llama2-7b-chat" in available_models else 0,
                help="Select an open-source LLM for insurance tasks"
            )

        # Display model details
        if model_id in model_details:
            details = model_details[model_id]

            # Highlight if model is CPU-friendly
            cpu_friendly_badge = "✅ CPU-Friendly" if details.get(
                "cpu_friendly", False) else "⚠️ May be slow on CPU"

            st.info(f"**{model_id}**\n\n"
                    f"Description: {details['description']}\n\n"
                    f"Parameters: {details['parameters']}\n\n"
                    f"Context Length: {details['context_length']}\n\n"
                    f"Best for: {details['suitable_for']}\n\n"
                    f"CPU Performance: {cpu_friendly_badge}")

        # CPU optimization checkbox
        cpu_optimize = st.checkbox(
            "Enable CPU optimizations",
            value=not has_cuda,  # Default checked when no GPU
            help="Apply special optimizations for CPU-only inference (recommended when running without GPU)"
        )

        # Change default quantization based on system capabilities and model
        default_quantization = 0  # 4bit by default

        # For CPU, use 8-bit for large models and no quantization for small models
        if not has_cuda:
            if model_id in ["phi-1.5", "phi-2", "tiny-llama-1b"]:
                default_quantization = 2  # None for small models on CPU
            else:
                default_quantization = 1  # 8bit for larger models
        elif has_cuda and free_memory_gb < 8:
            default_quantization = 1  # 8bit for low GPU memory

        quantization = st.selectbox(
            "Quantization",
            ["4bit", "8bit", None],
            index=default_quantization,
            help="Lower precision reduces memory requirements but may affect quality. For CPU, 8-bit works best for large models, while no quantization is better for smaller models like Phi-2."
        )

        # Add timeout setting option - longer default for CPU
        default_timeout = 300 if not has_cuda else 60
        generation_timeout = st.slider(
            "Generation Timeout (seconds)",
            min_value=30,
            max_value=600,
            value=default_timeout,
            step=30,
            help="Maximum time allowed for text generation before timeout. For CPU, use higher values (300+ seconds recommended)."
        )

        st.session_state.generation_timeout = generation_timeout

        # Add CPU-specific max tokens limit
        if not has_cuda:
            default_max_tokens = 256 if model_id not in cpu_friendly_models else 512
            max_tokens_limit = st.slider(
                "Default Max Tokens Limit",
                min_value=64,
                max_value=1024,
                value=default_max_tokens,
                step=64,
                help="Default limit for maximum tokens to generate. Lower values speed up generation on CPU."
            )
            st.session_state.default_max_tokens = max_tokens_limit

        return model_id, quantization, cpu_optimize

    def _load_model(self, model_id: str, quantization: str, cpu_optimize: bool):
        """Load the selected model."""
        try:
            # Determine device map
            device_map = "cpu" if cpu_optimize else "auto"

            # Memory optimization
            if self.torch_utils.is_cuda_available():
                self.torch_utils.clear_gpu_memory()

            # For CPU, show warning about loading time for large models
            has_cuda = self.torch_utils.is_cuda_available()
            cpu_friendly_models = get_cpu_friendly_models()

            if not has_cuda and model_id not in cpu_friendly_models:
                loading_message = st.empty()
                loading_message.warning(
                    f"Loading {model_id} on CPU. This may take several minutes. Please be patient...")

            # Load model with potential CPU optimization
            loading_kwargs = {
                "cpu_optimize": cpu_optimize
            }

            model, tokenizer, is_cpu_optimized = load_model(
                model_id=model_id,
                quantization=quantization,
                device_map=device_map,
                **loading_kwargs
            )

            # Set up the inference engine with CPU optimization flag
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.inference_engine = ModelInference(
                model, tokenizer, cpu_optimized=is_cpu_optimized
            )
            st.session_state.model_id = model_id
            st.session_state.quantization = quantization
            st.session_state.is_cpu_optimized = is_cpu_optimized

            st.success(f"Model {model_id} loaded successfully!")

            # Add guidance for CPU users
            if not has_cuda:
                st.info(
                    "💡 CPU Usage Tips: Keep prompts short, use small max token values, and be patient during generation. The first generation after loading will be the slowest.")

            # Automatically navigate to prompt engineering page
            SessionState.set_tab("prompt_engineering")
            st.rerun()

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            logger.error(f"Error loading model: {str(e)}")
            st.info(
                "Troubleshooting tips: Try using a smaller model or increase quantization to 8-bit. Make sure you have enough memory available.")

    def render(self):
        """Render the model selection page."""
        st.title("Model Selection")

        with st.expander("About Model Selection", expanded=False):
            st.markdown("""
            This page allows you to select and load an open-source LLM for insurance tasks.

            You can choose from various supported models and configure parameters such as
            quantization level to balance between performance and resource usage.
            
            If you're experiencing the app getting stuck during generation:
            - Try a smaller model (Phi-2 or TinyLLaMA are best for CPU)
            - Use 8bit quantization instead of 4bit
            - Reduce the max tokens in the generation settings
            - Use a shorter prompt
            - Increase the timeout value
            
            CPU-specific recommendations:
            - Phi-2 (2.7B parameters) offers the best balance of quality and speed
            - TinyLLaMA (1.1B parameters) is the fastest option
            - Avoid models larger than 7B parameters on CPU
            """)

        col1, col2 = st.columns([2, 1])

        with col1:
            with st.container():
                model_id, quantization, cpu_optimize = self._display_model_selection_ui()

            # Load model button
            if st.button("Load Model"):
                with st.spinner(f"Loading {model_id}..."):
                    self._load_model(model_id, quantization, cpu_optimize)

        with col2:
            self._display_system_info()


class PromptEngineeringPage:
    """Class to handle the prompt engineering page UI and logic."""

    def render(self):
        """Render the prompt engineering page."""
        st.header("Prompt Engineering")

        if st.session_state.inference_engine is None:
            st.warning("Please load a model first.")
            if st.button("Go to Model Selection"):
                SessionState.set_tab("model_selection")
                st.rerun()
            return

        # Get prompt library
        prompt_library = get_prompt_library()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Prompt Template Selection")

            # Get task types
            task_types = prompt_library.list_task_types()
            task_type = st.selectbox("Select Task Type", task_types)

            # Get templates for the selected task
            templates = prompt_library.get_templates_by_task(task_type)
            template_names = [t.name for t in templates]
            selected_template_name = st.selectbox(
                "Select Template", template_names)

            # Get the selected template
            selected_template = prompt_library.get_template(
                selected_template_name)

            if selected_template:
                st.text_area("Template Description",
                             selected_template.description, height=100)
                st.text_area(
                    "Template", selected_template.template, height=300)

                # Display required variables
                st.subheader("Input Variables")

                # Initialize variables dictionary
                variables = {}

                for var in selected_template.variables:
                    if var not in ["examples", "cot_examples", "reasoning_request"]:
                        if var == "policy_text":
                            st.write(f"**{var}**")

                            # Initialize the session state for this variable if it doesn't exist
                            if f"sample_{var}" not in st.session_state:
                                st.session_state[f"sample_{var}"] = False

                            # Create a unique key for this button
                            sample_data_button = st.button(
                                "Load Sample Policy", key=f"load_sample_{var}")

                            # When button is clicked, set the session state flag and load the data
                            if sample_data_button:
                                st.session_state[f"sample_{var}"] = True
                                st.session_state[f"{var}_content"] = DataLoader.load_sample_data(
                                    "policy")
                                st.rerun()

                            # If the session state flag is set, use the loaded content
                            if st.session_state.get(f"sample_{var}", False):
                                variables[var] = st.session_state.get(
                                    f"{var}_content", "")
                                text_area = st.text_area(
                                    f"Enter {var}", value=variables[var], height=200, key=f"{var}_textarea")
                                variables[var] = text_area
                            else:
                                variables[var] = st.text_area(
                                    f"Enter {var}", "", height=200, key=f"{var}_textarea")
                        elif var == "claim_text":
                            st.write(f"**{var}**")

                            # Initialize the session state for this variable if it doesn't exist
                            if f"sample_{var}" not in st.session_state:
                                st.session_state[f"sample_{var}"] = False

                            # Create a unique key for this button
                            sample_data_button = st.button(
                                "Load Sample Claim", key=f"load_sample_{var}")

                            # When button is clicked, set the session state flag and load the data
                            if sample_data_button:
                                st.session_state[f"sample_{var}"] = True
                                st.session_state[f"{var}_content"] = DataLoader.load_sample_data(
                                    "claim")
                                st.rerun()

                            # If the session state flag is set, use the loaded content
                            if st.session_state.get(f"sample_{var}", False):
                                variables[var] = st.session_state.get(
                                    f"{var}_content", "")
                                text_area = st.text_area(
                                    f"Enter {var}", value=variables[var], height=200, key=f"{var}_textarea")
                                variables[var] = text_area
                            else:
                                variables[var] = st.text_area(
                                    f"Enter {var}", "", height=200, key=f"{var}_textarea")
                        elif var == "inquiry_text":
                            st.write(f"**{var}**")

                            # Initialize the session state for this variable if it doesn't exist
                            if f"sample_{var}" not in st.session_state:
                                st.session_state[f"sample_{var}"] = False

                            # Create a unique key for this button
                            sample_data_button = st.button(
                                "Load Sample Inquiry", key=f"load_sample_{var}")

                            # When button is clicked, set the session state flag and load the data
                            if sample_data_button:
                                st.session_state[f"sample_{var}"] = True
                                st.session_state[f"{var}_content"] = DataLoader.load_sample_data(
                                    "customer_inquiry")
                                st.rerun()

                            # If the session state flag is set, use the loaded content
                            if st.session_state.get(f"sample_{var}", False):
                                variables[var] = st.session_state.get(
                                    f"{var}_content", "")
                                text_area = st.text_area(
                                    f"Enter {var}", value=variables[var], height=200, key=f"{var}_textarea")
                                variables[var] = text_area
                            else:
                                variables[var] = st.text_area(
                                    f"Enter {var}", "", height=200, key=f"{var}_textarea")
                        else:
                            variables[var] = st.text_area(
                                f"Enter {var}", "", height=200, key=f"{var}_textarea")

                # Prompt strategy selection
                st.subheader("Prompt Strategy")
                strategy_type = st.selectbox(
                    "Select Prompt Strategy",
                    ["zero_shot", "few_shot", "chain_of_thought"],
                    index=0 if selected_template.strategy_type == "zero_shot" else (
                        1 if selected_template.strategy_type == "few_shot" else 2
                    )
                )

        with col2:
            st.subheader("Generation Settings")

            # Model parameters
            # Get default max tokens from session state if running on CPU
            default_max_tokens = st.session_state.get(
                "default_max_tokens", 512)

            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                                    help="Higher values make output more random, lower values more deterministic")
            max_tokens = st.slider("Max Tokens", 64, 4096, default_max_tokens, 64,
                                   help="Maximum number of tokens to generate. Lower values are faster, especially on CPU.")

            # CPU optimization notice
            is_cpu_optimized = st.session_state.get("is_cpu_optimized", False)
            if is_cpu_optimized:
                st.info(
                    "ℹ️ Running with CPU optimizations. Generation will be slower but more stable.")

                # For CPU, add option for greedy decoding
                use_greedy = st.checkbox(
                    "Use greedy decoding",
                    value=False,
                    help="Faster but less creative text generation. Recommended for CPU."
                )

                # Additional warning for CPU users
                st.warning(
                    "⚠️ CPU generation can take several minutes. For best results, keep your prompts short and max tokens small.")

            # Generation button
            generate_button = st.button("Generate Output")

            if generate_button:
                # Check if all required variables are provided
                if all(v.strip() for v in variables.values()):
                    try:
                        # Create strategy and prepare generation
                        with st.spinner("Preparing prompt..."):
                            # Create prompt strategy
                            strategy_kwargs = {}
                            if strategy_type == "few_shot":
                                # Simple example for few-shot learning
                                if task_type == "policy_summary":
                                    strategy_kwargs = {
                                        "examples": [
                                            {
                                                "input": "Auto insurance policy for driver with $100k/$300k liability, $500 deductible",
                                                "output": "This policy provides auto insurance with bodily injury liability limits of $100,000 per person and $300,000 per accident. The deductible for physical damage is $500."
                                            }
                                        ],
                                        "example_template": "Input: {input}\nOutput: {output}"
                                    }
                                elif task_type == "claim_response":
                                    strategy_kwargs = {
                                        "examples": [
                                            {
                                                "claim": "Rear-end collision with damage to bumper",
                                                "response": "We have approved your claim for the rear-end collision. Your policy covers the damage to your bumper, less your deductible."
                                            }
                                        ],
                                        "example_template": "Claim: {claim}\nResponse: {response}"
                                    }
                            elif strategy_type == "chain_of_thought":
                                # Simple example for chain-of-thought reasoning
                                if task_type == "policy_summary":
                                    strategy_kwargs = {
                                        "cot_examples": [
                                            {
                                                "input": "Auto insurance policy with $100k/$300k liability, $500 deductible",
                                                "reasoning": "This policy has liability limits of $100k per person and $300k per accident. It also has a $500 deductible for physical damage.",
                                                "output": "This policy provides auto insurance with bodily injury liability limits of $100,000 per person and $300,000 per accident. The deductible for physical damage is $500."
                                            }
                                        ],
                                        "cot_example_template": "Input: {input}\nReasoning: {reasoning}\nOutput: {output}"
                                    }

                            strategy = create_prompt_strategy(
                                strategy_type, **strategy_kwargs)

                            # Apply prompt strategy to format the input
                            if strategy_type == "zero_shot":
                                formatted_prompt = selected_template.template.format(
                                    **variables)
                            else:
                                formatted_prompt = strategy.apply(
                                    variables, selected_template.template)

                            # Generate text with CPU optimizations if enabled
                            generation_params = {
                                "max_length": max_tokens,
                                "temperature": temperature,
                                "top_p": 0.9,
                                "top_k": 50,
                                "num_return_sequences": 1,
                                # Use greedy decoding if on CPU and selected
                                "do_sample": not (is_cpu_optimized and locals().get('use_greedy', False)),
                                # Get timeout from session state
                                "timeout_seconds": st.session_state.get("generation_timeout", 60)
                            }

                            # For CPU optimization, adjust parameters
                            if is_cpu_optimized:
                                if locals().get('use_greedy', False):
                                    # Greedy decoding for faster CPU inference
                                    generation_params["temperature"] = 1.0
                                    generation_params["top_p"] = 1.0
                                    generation_params["top_k"] = 1

                        # Now do the actual generation (outside the first spinner)
                        with st.spinner("Generating text... Please wait."):
                            # Add extra info for CPU users
                            if is_cpu_optimized:
                                st.info(
                                    f"Generating with timeout of {generation_params['timeout_seconds']} seconds. CPU generation may take a while...")

                            # Add a Streamlit progress indicator
                            progress_placeholder = st.empty()
                            progress_bar = progress_placeholder.progress(0.0)
                            status_text = st.empty()

                            # Run actual generation
                            status_text.text(
                                "Running model inference... (this may take a while on CPU)")

                            # Manual progress indication for CPU mode
                            if is_cpu_optimized:
                                # Just set progress at 50% and provide clear message
                                progress_bar.progress(0.5)
                                status_text.text(
                                    "Generating text... This may take several minutes on CPU. Please be patient.")

                            try:
                                # Directly run the generation
                                generation_start_time = time.time()

                                # Create a container for streaming output
                                output_container = st.empty()
                                generated_text = ""

                                # Define callback function for streaming
                                def streaming_callback(token):
                                    nonlocal generated_text
                                    generated_text += token
                                    output_container.markdown(
                                        f"**Generated Output:**\n\n{generated_text}")

                                # Run the streaming generation
                                full_text = st.session_state.inference_engine.generate_with_streaming(
                                    formatted_prompt,
                                    callback=streaming_callback,
                                    max_length=generation_params["max_length"],
                                    temperature=generation_params["temperature"],
                                    top_p=generation_params["top_p"],
                                    top_k=generation_params["top_k"],
                                    do_sample=generation_params["do_sample"],
                                    timeout_seconds=generation_params["timeout_seconds"]
                                )

                                # Generation complete - set progress to 100%
                                progress_bar.progress(1.0)

                                # Calculate elapsed time
                                elapsed_time = time.time() - generation_start_time
                                status_text.text(
                                    f"Generation completed in {elapsed_time:.2f} seconds")

                                # Clear progress indicators after short delay
                                # Brief pause to show completion
                                time.sleep(0.5)
                                progress_placeholder.empty()
                                status_text.empty()

                                # Check if it's an error message (returned by our custom timeout handler)
                                if full_text.startswith("Generation timed out") or full_text.startswith("Generation error"):
                                    st.error(full_text)
                                    st.info(
                                        "Try reducing the max tokens, using a simpler prompt, or increasing the timeout setting in the model selection page.")

                                    # CPU-specific suggestions
                                    if is_cpu_optimized:
                                        st.warning(
                                            "CPU-specific suggestions:")
                                        st.markdown("""
                                        - Try enabling 'greedy decoding' for faster generation
                                        - Use a much shorter prompt (under 100 words)
                                        - Reduce max tokens to 128 or less
                                        - Try a smaller model like Phi-2 or TinyLLaMA
                                        """)
                                else:
                                    # Save the generated text to session state
                                    if "generated_outputs" not in st.session_state:
                                        st.session_state.generated_outputs = []

                                    generation_time = time.time() - generation_start_time
                                    st.info(
                                        f"Generation completed in {generation_time:.2f} seconds")

                                    # Save the generated output
                                    st.session_state.generated_outputs.append({
                                        "template_name": selected_template_name,
                                        "task_type": task_type,
                                        "strategy_type": strategy_type,
                                        "input_variables": variables.copy(),
                                        "generation_params": generation_params.copy(),
                                        "output": full_text,
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "generation_time": generation_time
                                    })

                                    # Show evaluation button
                                    if st.button("Evaluate Output"):
                                        SessionState.set_tab("evaluation")
                                        st.rerun()

                            except Exception as e:
                                st.error(f"Error during generation: {str(e)}")
                                logger.error(
                                    f"Error during generation: {str(e)}")
                                st.info(
                                    "Troubleshooting tips: Try clearing your browser cache, restarting the app, or using a smaller model.")

                    except Exception as e:
                        st.error(f"Error preparing generation: {str(e)}")
                        logger.error(f"Error preparing generation: {str(e)}")
                else:
                    st.error("Please fill in all required variables.")

            # Show previous outputs
            if st.session_state.generated_outputs:
                st.subheader("Previous Outputs")

                for idx, output in enumerate(st.session_state.generated_outputs):
                    with st.expander(f"Output {idx+1}: {output['template_name']} ({output['timestamp']})"):
                        st.write(f"Task Type: {output['task_type']}")
                        st.write(f"Strategy: {output['strategy_type']}")
                        st.text_area(
                            f"Generated Text {idx+1}", output['output'], height=150)

                        # Option to evaluate this output
                        if st.button(f"Evaluate Output {idx+1}"):
                            # Set the output to evaluate
                            st.session_state.current_output_idx = idx
                            SessionState.set_tab("evaluation")
                            st.rerun()


class EvaluationPage:
    """Class to handle the evaluation page UI and logic."""

    def render(self):
        """Render the evaluation page."""
        st.header("Evaluation")

        if not st.session_state.generated_outputs:
            st.warning("No outputs to evaluate. Generate some outputs first.")
            if st.button("Go to Prompt Engineering"):
                SessionState.set_tab("prompt_engineering")
                st.rerun()
            return

        # Get metrics manager
        metrics_manager = get_metrics_manager()

        # Current output to evaluate
        current_idx = getattr(st.session_state, "current_output_idx", len(
            st.session_state.generated_outputs) - 1)
        current_output = st.session_state.generated_outputs[current_idx]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Output to Evaluate")
            st.write(f"Task Type: {current_output['task_type']}")
            st.write(f"Template: {current_output['template_name']}")
            st.write(f"Strategy: {current_output['strategy_type']}")

            # Display the input variables and output
            st.text_area("Input Variables", str(
                current_output['input_variables']), height=150)
            st.text_area("Generated Output",
                         current_output['output'], height=300)

            # Reference text input for comparison-based metrics
            st.subheader("Reference Text (Optional)")
            reference_text = st.text_area("Enter reference text for comparison", "", height=200,
                                          help="Provide a reference text to compare the generated output against")

        with col2:
            st.subheader("Evaluation Metrics")

            # Select metrics to use
            available_metrics = metrics_manager.list_metrics()
            selected_metrics = st.multiselect(
                "Select metrics to use",
                [m["name"] for m in available_metrics],
                default=["relevance", "completeness", "complexity"]
            )

            # Additional context for evaluation
            st.subheader("Additional Context")

            # Context based on task type
            context = {}

            if current_output['task_type'] == "policy_summary":
                required_sections = st.text_input(
                    "Required Sections (comma-separated)",
                    "coverages,limits,exclusions,premium",
                    help="Sections that should be included in the summary"
                )
                context["required_sections"] = [s.strip()
                                                for s in required_sections.split(",")]

                required_phrases = st.text_input(
                    "Required Phrases (comma-separated)",
                    "policy,coverage,deductible",
                    help="Phrases that should be included in the output"
                )
                context["required_phrases"] = [s.strip()
                                               for s in required_phrases.split(",")]

                prohibited_phrases = st.text_input(
                    "Prohibited Phrases (comma-separated)",
                    "uncertain,unclear,not mentioned",
                    help="Phrases that should not be included in the output"
                )
                context["prohibited_phrases"] = [s.strip()
                                                 for s in prohibited_phrases.split(",")]

            elif current_output['task_type'] == "claim_response":
                required_phrases = st.text_input(
                    "Required Phrases (comma-separated)",
                    "claim,policy,coverage",
                    help="Phrases that should be included in the output"
                )
                context["required_phrases"] = [s.strip()
                                               for s in required_phrases.split(",")]

                prohibited_phrases = st.text_input(
                    "Prohibited Phrases (comma-separated)",
                    "uncertain,unclear,not mentioned",
                    help="Phrases that should not be included in the output"
                )
                context["prohibited_phrases"] = [s.strip()
                                                 for s in prohibited_phrases.split(",")]

            # Evaluate button
            if st.button("Run Evaluation"):
                with st.spinner("Evaluating..."):
                    try:
                        # Run evaluation with selected metrics
                        evaluation_results = metrics_manager.evaluate(
                            generated_text=current_output['output'],
                            metric_names=selected_metrics,
                            reference_text=reference_text if reference_text.strip() else None,
                            context=context
                        )

                        # Store results
                        st.session_state.evaluation_results[current_idx] = {
                            "metrics": [result.as_dict() for result in evaluation_results.values()],
                            "reference_text": reference_text,
                            "context": context
                        }

                        # Show success message
                        st.success("Evaluation complete!")

                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        logger.error(f"Error during evaluation: {str(e)}")

            # Display evaluation results if available
            if current_idx in st.session_state.evaluation_results:
                st.subheader("Evaluation Results")

                results = st.session_state.evaluation_results[current_idx]

                # Create a DataFrame for the results
                results_df = pd.DataFrame([
                    {
                        "Metric": result["metric_name"],
                        "Score": result["score"],
                        "Max Score": result["max_score"],
                        "Normalized Score": result["normalized_score"]
                    }
                    for result in results["metrics"]
                ])

                # Display the results table
                results_df = prepare_dataframe_for_display(results_df)
                st.dataframe(results_df)

                # Create a bar chart of the normalized scores
                fig = px.bar(
                    results_df,
                    x="Metric",
                    y="Normalized Score",
                    title="Evaluation Scores",
                    color="Normalized Score",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 1]
                )
                st.plotly_chart(fig)

                # Show detailed results in expanders
                for result in results["metrics"]:
                    with st.expander(f"Detailed Results for {result['metric_name']}"):
                        st.write(
                            f"Score: {result['score']:.3f} / {result['max_score']:.3f}")
                        st.write(
                            f"Normalized Score: {result['normalized_score']:.3f}")

                        if result["details"]:
                            st.write("Details:")
                            st.json(result["details"])

                # Export results button
                if st.button("Export Results"):
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    export_path = f"evaluation_results_{timestamp}.json"

                    try:
                        with open(export_path, "w") as f:
                            json.dump({
                                "output": current_output,
                                "evaluation_results": results
                            }, f, indent=2)

                        st.success(f"Results exported to {export_path}")

                    except Exception as e:
                        st.error(f"Error exporting results: {str(e)}")
                        logger.error(f"Error exporting results: {str(e)}")


class BenchmarksPage:
    """Class to handle the benchmarks page UI and logic."""

    def render(self):
        """Render the benchmarks page."""
        st.header("Benchmarks")

        if st.session_state.inference_engine is None:
            st.warning("Please load a model first.")
            if st.button("Go to Model Selection"):
                SessionState.set_tab("model_selection")
                st.rerun()
            return

        # Get benchmark manager
        benchmark_manager = get_benchmark_manager()

        # Load or create sample benchmarks if they don't exist
        if not benchmark_manager.list_benchmarks():
            # Add debug logging
            logger.info(
                f"No benchmarks found in directory: {benchmark_manager.benchmarks_dir}")

            if st.button("Create Sample Benchmarks"):
                with st.spinner("Creating sample benchmarks..."):
                    try:
                        logger.info("Creating sample benchmarks...")

                        # Ensure the benchmark directory exists
                        os.makedirs(
                            benchmark_manager.benchmarks_dir, exist_ok=True)
                        logger.info(
                            f"Ensured benchmark directory exists: {benchmark_manager.benchmarks_dir}")

                        # Create policy summary benchmark
                        policy_benchmark = create_policy_summary_benchmark()
                        if policy_benchmark is None:
                            raise ValueError(
                                "Failed to create policy summary benchmark")
                        logger.info(
                            f"Created policy benchmark: {policy_benchmark.name} with {len(policy_benchmark.examples)} examples")
                        benchmark_manager.benchmarks[policy_benchmark.name] = policy_benchmark

                        # Save the benchmark to disk
                        benchmark_path = os.path.join(
                            benchmark_manager.benchmarks_dir, f"{policy_benchmark.name}.json")
                        policy_benchmark.save(benchmark_path)
                        logger.info(
                            f"Saved policy benchmark to {benchmark_path}")

                        # Create claim response benchmark
                        claim_benchmark = create_claim_response_benchmark()
                        if claim_benchmark is None:
                            raise ValueError(
                                "Failed to create claim response benchmark")
                        logger.info(
                            f"Created claim benchmark: {claim_benchmark.name} with {len(claim_benchmark.examples)} examples")
                        benchmark_manager.benchmarks[claim_benchmark.name] = claim_benchmark

                        # Save the benchmark to disk
                        benchmark_path = os.path.join(
                            benchmark_manager.benchmarks_dir, f"{claim_benchmark.name}.json")
                        claim_benchmark.save(benchmark_path)
                        logger.info(
                            f"Saved claim benchmark to {benchmark_path}")

                        st.success("Sample benchmarks created successfully!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error creating sample benchmarks: {str(e)}")
                        logger.error(
                            f"Error creating sample benchmarks: {str(e)}")
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Available Benchmarks")

                # List available benchmarks
                benchmarks = benchmark_manager.list_benchmarks()
                benchmark_names = [b["name"] for b in benchmarks]
                selected_benchmark = st.selectbox(
                    "Select Benchmark", benchmark_names)

                # Display benchmark details
                if selected_benchmark:
                    benchmark = benchmark_manager.get_benchmark(
                        selected_benchmark)

                    if benchmark:
                        st.write(f"Task Type: {benchmark.task_type}")
                        st.write(f"Description: {benchmark.description}")
                        st.write(
                            f"Number of Examples: {len(benchmark.examples)}")
                        st.write(f"Metrics: {', '.join(benchmark.metrics)}")

                        # Option to view examples
                        with st.expander("View Examples"):
                            for idx, example in enumerate(benchmark.examples):
                                st.write(f"Example {idx+1}: {example.id}")
                                st.text_area(
                                    f"Input {idx+1}", example.input_text, height=150)
                                st.text_area(
                                    f"Reference Output {idx+1}", example.reference_output, height=150)

                        # Run benchmark button
                        if st.button("Run Benchmark"):
                            with st.spinner(f"Running benchmark {selected_benchmark}..."):
                                try:
                                    # Create a function to generate output
                                    def generate_output(input_text):
                                        generation_params = {
                                            "max_length": 512,
                                            "temperature": 0.7,
                                            "top_p": 0.9,
                                            "top_k": 50,
                                            "do_sample": True
                                        }

                                        # For streaming display during benchmark runs
                                        if st.checkbox("Show generation in real-time", value=True):
                                            # Create a container for streaming output
                                            output_container = st.empty()
                                            generated_text = ""

                                            # Define callback function for streaming
                                            def streaming_callback(token):
                                                nonlocal generated_text
                                                generated_text += token
                                                output_container.markdown(
                                                    f"**Generating:**\n\n{generated_text}")

                                            # Run the streaming generation
                                            full_text = st.session_state.inference_engine.generate_with_streaming(
                                                input_text,
                                                callback=streaming_callback,
                                                max_length=generation_params["max_length"],
                                                temperature=generation_params["temperature"],
                                                top_p=generation_params["top_p"],
                                                top_k=generation_params["top_k"],
                                                do_sample=generation_params["do_sample"]
                                            )

                                            return full_text
                                        else:
                                            # Use regular generation if streaming is not selected
                                            generated_texts = st.session_state.inference_engine.generate(
                                                input_text,
                                                **generation_params
                                            )

                                            return generated_texts[0] if generated_texts else ""

                                    # Run the benchmark
                                    model_id = st.session_state.model.__class__.__name__
                                    result = benchmark_manager.run_benchmark(
                                        benchmark_name=selected_benchmark,
                                        model_id=model_id,
                                        generate_fn=generate_output
                                    )

                                    # Store result in session state
                                    if "benchmark_results" not in st.session_state:
                                        st.session_state.benchmark_results = {}

                                    st.session_state.benchmark_results[selected_benchmark] = result

                                    st.success(
                                        f"Benchmark {selected_benchmark} completed successfully!")

                                except Exception as e:
                                    st.error(
                                        f"Error running benchmark: {str(e)}")
                                    logger.error(
                                        f"Error running benchmark: {str(e)}")

            with col2:
                st.subheader("Benchmark Results")

                # Display benchmark results if available
                if hasattr(st.session_state, "benchmark_results") and st.session_state.benchmark_results:
                    # Select result to display
                    result_keys = list(
                        st.session_state.benchmark_results.keys())
                    selected_result = st.selectbox(
                        "Select Result", result_keys)

                    if selected_result:
                        result = st.session_state.benchmark_results[selected_result]

                        # Display aggregate scores
                        st.write("Aggregate Scores:")

                        # Create a DataFrame for the aggregate scores
                        scores_df = pd.DataFrame([
                            {"Metric": metric, "Score": score}
                            for metric, score in result.aggregate_scores.items()
                        ])

                        # Display the scores table
                        scores_df = prepare_dataframe_for_display(scores_df)
                        st.dataframe(scores_df)

                        # Create a radar chart of the scores
                        metrics = [
                            m for m in result.aggregate_scores.keys() if m != "overall"]
                        scores = [result.aggregate_scores[m] for m in metrics]

                        # Radar chart
                        fig = go.Figure()

                        fig.add_trace(go.Scatterpolar(
                            r=scores,
                            theta=metrics,
                            fill='toself',
                            name=result.model_id
                        ))

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title=f"Benchmark Results: {selected_result}"
                        )

                        st.plotly_chart(fig)

                        # Export button
                        if st.button("Export Benchmark Results"):
                            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                            export_path = f"benchmark_results_{selected_result}_{timestamp}.json"

                            try:
                                with open(export_path, "w") as f:
                                    json.dump(result.to_dict(), f, indent=2)

                                st.success(
                                    f"Results exported to {export_path}")

                            except Exception as e:
                                st.error(f"Error exporting results: {str(e)}")
                                logger.error(
                                    f"Error exporting results: {str(e)}")

                        # Detailed results
                        with st.expander("View Detailed Results"):
                            for idx, example_result in enumerate(result.results):
                                st.write(
                                    f"Example {idx+1}: {example_result['example_id']}")

                                col3, col4 = st.columns(2)

                                with col3:
                                    st.text_area(
                                        f"Generated Output {idx+1}", example_result['generated_text'], height=150)

                                with col4:
                                    st.text_area(
                                        f"Reference Output {idx+1}", example_result['reference_output'], height=150)

                                # Metrics for this example
                                st.write("Metrics:")
                                for metric, score in example_result['metrics'].items():
                                    st.write(f"{metric}: {score:.3f}")
                else:
                    st.info(
                        "No benchmark results available. Run a benchmark first.")


class ModelComparisonPage:
    """Class to handle the model comparison page UI and logic."""

    def render(self):
        """Render the model comparison page."""
        st.header("Model Comparison")

        # Check if there are benchmark results to compare
        if not hasattr(st.session_state, "benchmark_results") or not st.session_state.benchmark_results:
            st.warning("No benchmark results available. Run benchmarks first.")
            if st.button("Go to Benchmarks"):
                SessionState.set_tab("benchmarks")
                st.rerun()
            return

        # Get benchmark manager
        benchmark_manager = get_benchmark_manager()

        # List available benchmark results
        result_keys = list(st.session_state.benchmark_results.keys())

        st.subheader("Compare Models")
        st.write("Load more models and run benchmarks to compare their performance.")

        # Display comparison if multiple results exist
        if len(result_keys) > 1:
            # Select benchmark to compare
            selected_benchmark = st.selectbox("Select Benchmark", result_keys)

            # Get results for the selected benchmark
            results = {
                result.model_id: result
                for key, result in st.session_state.benchmark_results.items()
                if key == selected_benchmark
            }

            # Compare models
            comparison_df = benchmark_manager.compare_models(
                selected_benchmark, results)

            # Display comparison
            st.subheader(f"Comparison for {selected_benchmark}")
            comparison_df = prepare_dataframe_for_display(comparison_df)
            st.dataframe(comparison_df)

            # Create comparison charts
            metrics = [
                col for col in comparison_df.columns if col.endswith("_score")]

            if metrics:
                # Bar chart for overall score
                fig1 = px.bar(
                    comparison_df,
                    x="model_id",
                    y="overall_score",
                    title="Overall Score Comparison",
                    color="overall_score",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 1]
                )
                st.plotly_chart(fig1)

                # Bar chart for individual metrics
                fig2 = px.bar(
                    comparison_df.melt(
                        id_vars=["model_id"],
                        value_vars=metrics,
                        var_name="Metric",
                        value_name="Score"
                    ),
                    x="model_id",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Metric Score Comparison"
                )
                st.plotly_chart(fig2)

                # Radar chart comparison
                metric_names = [m.replace("_score", "") for m in metrics]

                fig3 = go.Figure()

                for model_id in comparison_df["model_id"]:
                    model_data = comparison_df[comparison_df["model_id"] == model_id]
                    scores = [model_data[m].values[0] for m in metrics]

                    fig3.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=metric_names,
                        fill='toself',
                        name=model_id
                    ))

                fig3.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Model Comparison Radar Chart"
                )

                st.plotly_chart(fig3)
        else:
            st.info(
                "Run benchmarks with at least two different models to enable comparison.")


class SettingsPage:
    """Class to handle the settings page UI and logic."""

    def render(self):
        """Render the settings page."""
        st.header("Settings")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Application Settings")

            # Theme setting
            theme = st.selectbox(
                "Application Theme",
                ["Light", "Dark"],
                index=1
            )

            # Cache setting
            cache_models = st.checkbox("Cache Models", value=True,
                                       help="Keep models in memory between sessions")

            # Debug mode
            debug_mode = st.checkbox("Debug Mode", value=False,
                                     help="Show additional debugging information")

            # Save settings button
            if st.button("Save Settings"):
                st.success("Settings saved successfully!")

        with col2:
            st.subheader("System Information")

            # Display system information
            torch = TorchUtils.get_torch()
            system_info = {
                "Python Version": os.sys.version.split()[0],
                "Streamlit Version": st.__version__,
                "PyTorch Version": torch.__version__,
                "CUDA Available": torch.cuda.is_available(),
                "GPU Count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "Current Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }

            for key, value in system_info.items():
                st.text(f"{key}: {value}")

            # Memory usage for loaded model
            if st.session_state.model is not None:
                mem_usage = "Unknown"
                try:
                    if hasattr(torch.cuda, "memory_allocated"):
                        mem_usage = f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
                except:
                    pass

                st.text(f"Model Memory Usage: {mem_usage}")

            # Clear session button
            if st.button("Clear Session"):
                # Reset session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]

                st.success("Session cleared successfully!")
                st.rerun()


def main():
    """Main function to run the Streamlit app."""
    # Set page configuration
    st.set_page_config(
        page_title="Insurance LLM Framework",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    SessionState.initialize()

    # Sidebar navigation
    st.sidebar.title("Insurance LLM Framework")
    st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/3448/3448502.png", width=100)

    # Navigation options
    nav_options = {
        "Model Selection": "model_selection",
        "Prompt Engineering": "prompt_engineering",
        "Evaluation": "evaluation",
        "Benchmarks": "benchmarks",
        "Model Comparison": "model_comparison",
        "Settings": "settings"
    }

    for label, tab in nav_options.items():
        if st.sidebar.button(label, key=f"nav_{tab}"):
            SessionState.set_tab(tab)
            st.rerun()

    # Render the selected page
    current_tab = st.session_state.current_tab

    if current_tab == "model_selection":
        model_selection_page = ModelSelectionPage()
        model_selection_page.render()
    elif current_tab == "prompt_engineering":
        prompt_engineering_page = PromptEngineeringPage()
        prompt_engineering_page.render()
    elif current_tab == "evaluation":
        evaluation_page = EvaluationPage()
        evaluation_page.render()
    elif current_tab == "benchmarks":
        benchmarks_page = BenchmarksPage()
        benchmarks_page.render()
    elif current_tab == "model_comparison":
        model_comparison_page = ModelComparisonPage()
        model_comparison_page.render()
    elif current_tab == "settings":
        settings_page = SettingsPage()
        settings_page.render()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Open-Source Prompt Engineering and Evaluation Framework for Insurance Domain Applications. "
        "This project helps insurance professionals leverage open-source LLMs for tasks such as "
        "policy summarization, claim processing, and customer communication."
    )


if __name__ == "__main__":
    main()
