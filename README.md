# Insurance LLM Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source prompt engineering and evaluation framework for insurance domain applications, leveraging the power of Large Language Models (LLMs) to transform insurance workflows.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Key Components](#key-components)
  - [Class Hierarchy](#class-hierarchy)
- [Model Support](#model-support)
- [CPU vs GPU Optimization](#cpu-vs-gpu-optimization)
- [Prompt Engineering](#prompt-engineering)
- [Evaluation Framework](#evaluation-framework)
- [Benchmarking](#benchmarking)
- [Extending the Framework](#extending-the-framework)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The Insurance LLM Framework provides a comprehensive suite of tools for insurance professionals to leverage open-source Large Language Models (LLMs) for various domain-specific tasks. By combining prompt engineering, model management, and evaluation capabilities, the framework enables insurance companies to harness the power of AI for improving operational efficiency and customer experience.

### Insurance Domain Applications

The framework is specifically designed for insurance-related tasks such as:

- **Policy Summarization**: Generate concise summaries of complex insurance policies
- **Claim Response Drafting**: Create professional responses to insurance claims
- **Risk Assessment Reporting**: Analyze and report on risk factors from unstructured data
- **Customer Communication**: Generate personalized customer communications
- **Underwriting Assistance**: Support underwriters with relevant information extraction
- **Compliance Checking**: Verify document compliance with regulatory requirements

## Key Features

The framework offers a comprehensive set of features designed to make LLMs accessible and effective for insurance professionals:

### Prompt Library
- Extensive collection of insurance-specific prompt templates
- Customizable templates with variable substitution
- Support for different prompting strategies (zero-shot, few-shot, chain-of-thought)
- Template management interface for creating and editing prompts

### Model Selection
- Support for multiple open-source LLMs (LLaMA-2, Mistral, Falcon, Phi-2, etc.)
- Optimized configurations for both CPU and GPU environments
- Quantization options for resource-constrained environments
- Model performance metrics and comparison tools

### Evaluation Dashboard
- Automated metrics for assessing output quality (ROUGE, BLEU, BERTScore)
- Human evaluation protocols with customizable rubrics
- Comparative evaluation across different models and prompts
- Visualization of evaluation results

### Output Management
- Save and export generated content in multiple formats
- Version tracking for generated outputs
- Batch processing capabilities for high-volume tasks
- Integration with common document formats

### Benchmarking
- Pre-defined benchmark datasets for insurance tasks
- Custom benchmark creation tools
- Performance comparison across models and configurations
- Detailed reporting and visualization

### Customization
- Extensible architecture for adding new models and capabilities
- API for programmatic access to framework components
- Configuration options for different deployment scenarios
- Support for custom evaluation metrics

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (for small models on CPU)
- **Storage**: 5GB for base installation, plus model storage (varies by model)
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+ or other Linux distributions

### Recommended Requirements
- **Python**: 3.10 or higher
- **RAM**: 16GB+ (32GB+ for larger models)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for faster inference)
- **CUDA**: 11.7 or higher (for GPU acceleration)
- **Storage**: 20GB+ SSD storage
- **OS**: Ubuntu 22.04 or other Linux distributions

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package installer)
- git (for cloning the repository)
- Virtual environment tool (venv, conda, etc.)

### Step-by-Step Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/insurance-llm-framework.git
cd insurance-llm-framework

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU dependencies if you have a compatible GPU
pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### Docker Installation (Alternative)

```bash
# Build the Docker image
docker build -t insurance-llm-framework .

# Run the container
docker run -p 8501:8501 -v $(pwd)/data:/app/data insurance-llm-framework
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# HuggingFace token for accessing gated models (required for some models)
HF_TOKEN=your_huggingface_token

# Application settings
APP_PORT=8501
APP_HOST=0.0.0.0

# Model cache directory (optional)
TRANSFORMERS_CACHE=./models/cache

# Logging level (optional)
LOG_LEVEL=INFO
```

## Usage

### Starting the Web Interface

```bash
# Basic usage
python run.py

# With custom port and host
python run.py --port 8502 --host 127.0.0.1

# With debug logging
LOG_LEVEL=DEBUG python run.py
```

### Web Interface Navigation

1. **Model Selection**: Choose and load an LLM model
2. **Prompt Engineering**: Create and test prompts for insurance tasks
3. **Evaluation**: Assess the quality of generated outputs
4. **Benchmarks**: Run and compare model performance on benchmark datasets
5. **Model Comparison**: Compare different models and configurations
6. **Settings**: Configure application settings and view system information


### API Usage

The framework components can be imported and used programmatically:

```python
from insurance_llm.models import ModelLoader
from insurance_llm.prompts import PromptLibrary
from insurance_llm.evaluation import EvaluationMetrics

# Load a model
model_loader = ModelLoader()
model, tokenizer = model_loader.load_model("phi-2", quantization="8bit")

# Get a prompt template
prompt_library = PromptLibrary()
template = prompt_library.get_template("policy_summary")

# Generate text
prompt = template.format(policy_text="Your policy text here...")
inference = ModelInference(model, tokenizer)
result = inference.generate(prompt, max_length=512)

# Evaluate the result
metrics = EvaluationMetrics()
score = metrics.evaluate(result, reference_text="Reference summary")
print(f"ROUGE-L score: {score['rouge-l']}")
```

## Project Structure

```
insurance-llm-framework/
├── app.py                      # Main Streamlit application
├── run.py                      # Application startup script
├── requirements.txt            # Project dependencies
├── Dockerfile                  # Docker configuration
├── .env                        # Environment variables (create this)
├── config/                     # Configuration files
│   ├── models.yaml             # Model configuration
│   ├── prompts.yaml            # Prompt configuration
│   └── evaluation.yaml         # Evaluation configuration
├── data/                       # Sample insurance data
│   ├── policies/               # Sample policy documents
│   ├── claims/                 # Sample claim documents
│   └── communications/         # Sample customer communications
├── models/                     # Model integration
│   ├── model_loader.py         # Classes for loading models
│   ├── inference.py            # Classes for model inference
│   └── cache/                  # Model cache directory
├── prompts/                    # Prompt engineering components
│   ├── templates/              # Reusable prompt templates
│   ├── strategies.py           # Prompt design strategies
│   └── library.py              # Prompt library manager
├── evaluation/                 # Evaluation components
│   ├── metrics.py              # Automated evaluation metrics
│   ├── human_eval.py           # Human evaluation protocols
│   ├── benchmarks.py           # Benchmark datasets and tests
│   ├── evaluations/            # Evaluation results
│   └── benchmarks/             # Benchmark datasets
├── ui/                         # UI components
│   ├── pages/                  # Different pages of the app
│   ├── components/             # Reusable UI components
│   └── utils.py                # UI utility functions
├── utils/                      # Utility functions
│   ├── logging.py              # Logging configuration
│   ├── file_utils.py           # File handling utilities
│   └── text_processing.py      # Text processing utilities
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test fixtures
└── docs/                       # Detailed documentation
    ├── use_cases.md            # Insurance use cases
    ├── prompt_engineering.md   # Prompt engineering guide
    ├── evaluation.md           # Evaluation guide
    ├── api_reference.md        # API documentation
    └── examples/               # Example notebooks and scripts
```

## Architecture
### Key Components

#### Model Management
- **ModelConfig**: Configuration for model repositories and settings
- **ModelLoader**: Handles loading and configuring LLM models
- **ModelInference**: Manages text generation with loaded models

#### Prompt Engineering
- **PromptTemplate**: Represents a single prompt template with variables
- **PromptLibrary**: Manages a collection of prompt templates
- **PromptStrategy**: Implements different prompting strategies

#### Evaluation
- **EvaluationMetric**: Base class for evaluation metrics
- **MetricsManager**: Manages and applies multiple evaluation metrics
- **HumanEvaluationManager**: Handles human evaluation workflows

#### Benchmarking
- **Benchmark**: Represents a benchmark dataset with examples
- **BenchmarkManager**: Manages benchmark datasets and runs tests
- **BenchmarkResult**: Stores and analyzes benchmark results

#### UI Components
- **ModelSelectionPage**: UI for selecting and loading models
- **PromptEngineeringPage**: UI for creating and testing prompts
- **EvaluationPage**: UI for evaluating generated outputs
- **BenchmarksPage**: UI for running and viewing benchmarks

#### Utilities
- **TorchUtils**: Utilities for PyTorch operations
- **SessionState**: Manages Streamlit session state
- **DataLoader**: Handles loading sample data
- **EnvironmentSetup**: Sets up the application environment

### Class Hierarchy

```
ModelManagement
├── ModelConfig
├── ModelLoader
├── PipelineFactory
└── ModelInference

PromptEngineering
├── PromptTemplate
├── PromptLibrary
└── PromptStrategy
    ├── ZeroShotStrategy
    ├── FewShotStrategy
    └── ChainOfThoughtStrategy

Evaluation
├── EvaluationMetric
│   ├── ROUGEMetric
│   ├── BLEUMetric
│   ├── BERTScoreMetric
│   └── CustomMetric
├── MetricsManager
└── HumanEvaluationManager

Benchmarking
├── Benchmark
├── BenchmarkExample
├── BenchmarkManager
└── BenchmarkResult

UI
├── ModelSelectionPage
├── PromptEngineeringPage
├── EvaluationPage
├── BenchmarksPage
├── ModelComparisonPage
└── SettingsPage

Utilities
├── TorchUtils
├── SessionState
├── DataLoader
├── ThreadingUtils
├── SystemInfo
└── EnvironmentSetup
```

## Model Support

The framework supports a variety of open-source LLMs with different capabilities and resource requirements:

### Supported Models

| Model | Parameters | Context Length | Best For | CPU Friendly |
|-------|------------|---------------|----------|--------------|
| LLaMA-2 7B | 7 billion | 4096 tokens | General text generation | No |
| LLaMA-2 13B | 13 billion | 4096 tokens | Higher quality generation | No |
| LLaMA-2 7B Chat | 7 billion | 4096 tokens | Conversational applications | No |
| LLaMA-2 13B Chat | 13 billion | 4096 tokens | Higher quality conversations | No |
| Mistral 7B | 7 billion | 8192 tokens | Long context generation | No |
| Mistral 7B Instruct | 7 billion | 8192 tokens | Instruction following | No |
| Falcon 7B | 7 billion | 2048 tokens | Efficient generation | No |
| Falcon 7B Instruct | 7 billion | 2048 tokens | Instruction following | No |
| Phi-2 | 2.7 billion | 2048 tokens | CPU-friendly generation | Yes |
| Phi-1.5 | 1.3 billion | 2048 tokens | Lightweight inference | Yes |
| TinyLLaMA 1.1B | 1.1 billion | 2048 tokens | Fast CPU inference | Yes |

### Model Selection Guidelines

- **For GPU environments**: 
  - LLaMA-2 13B Chat or Mistral 7B Instruct provide the best quality
  - Use 4-bit quantization to reduce VRAM requirements
  
- **For CPU environments**:
  - Phi-2 offers the best balance of quality and speed
  - TinyLLaMA 1.1B is the fastest option
  - Avoid models larger than 7B parameters

### Adding Custom Models

The framework supports adding custom models by extending the `ModelConfig` class:

```python
# Add a custom model to the configuration
ModelConfig.MODEL_REPOS["custom-model"] = "path/to/custom/model"
ModelConfig.MODEL_DETAILS["custom-model"] = {
    "description": "Custom model description",
    "parameters": "X billion",
    "context_length": "Y tokens",
    "suitable_for": "Specific tasks",
    "cpu_friendly": False
}
```

## 💻 CPU vs GPU Optimization

The framework includes extensive optimizations for both CPU and GPU environments:

### GPU Optimizations

- Automatic device mapping based on available GPU memory
- Quantization options (4-bit, 8-bit) to reduce VRAM requirements
- Batch processing for efficient multi-input generation
- Memory management to prevent CUDA out-of-memory errors

### CPU Optimizations

- Special handling for CPU-friendly models (Phi-2, TinyLLaMA)
- Automatic adjustment of generation parameters for better performance
- Reduced token generation limits to prevent timeouts
- Timeout management for long-running generations
- Single-thread processing for limited CPU environments

### Performance Comparison

| Model | GPU (RTX 3090) | CPU (8 cores) | CPU (4 cores) |
|-------|----------------|---------------|---------------|
| LLaMA-2 13B | 15 tokens/sec | 0.5 tokens/sec | 0.2 tokens/sec |
| LLaMA-2 7B | 30 tokens/sec | 1 token/sec | 0.5 tokens/sec |
| Mistral 7B | 25 tokens/sec | 0.8 tokens/sec | 0.4 tokens/sec |
| Phi-2 | 60 tokens/sec | 3 tokens/sec | 1.5 tokens/sec |
| TinyLLaMA 1.1B | 100 tokens/sec | 5 tokens/sec | 2.5 tokens/sec |

## Prompt Engineering

The framework provides comprehensive prompt engineering capabilities for insurance domain tasks:

### Prompt Templates

Prompt templates are structured with variables that can be substituted at runtime:

```
Template: policy_summary
Task: Summarize the key points of an insurance policy
Variables: policy_text
Strategy: zero_shot

I need to understand the key points of this insurance policy. Please provide a concise summary that includes:
1. Coverage limits
2. Major exclusions
3. Deductible amounts
4. Important conditions

Policy text:
{policy_text}

Summary:
```

### Prompt Strategies

The framework supports multiple prompting strategies:

- **Zero-Shot**: Direct prompting without examples
- **Few-Shot**: Including examples in the prompt
- **Chain-of-Thought**: Breaking down complex reasoning
- **ReAct**: Reasoning and acting iteratively

### Domain-Specific Templates

The framework includes templates for common insurance tasks:

- Policy summarization
- Claim response generation
- Risk assessment
- Customer inquiry handling
- Compliance checking
- Underwriting assistance

### Creating Custom Templates

Custom templates can be created through the UI or programmatically:

```python
from prompts.library import PromptTemplate, PromptLibrary

# Create a new template
template = PromptTemplate(
    name="custom_template",
    template="This is a template with {variable1} and {variable2}",
    task_type="custom_task",
    description="A custom template for specific tasks",
    variables=["variable1", "variable2"],
    strategy_type="zero_shot"
)

# Add to library
library = PromptLibrary()
library.add_template(template)
```

## Evaluation Framework

The framework provides comprehensive evaluation capabilities for assessing the quality of generated outputs:

### Automated Metrics

- **ROUGE**: Measures overlap between generated and reference texts
- **BLEU**: Evaluates translation quality
- **BERTScore**: Semantic similarity using BERT embeddings
- **Custom metrics**: Domain-specific metrics for insurance tasks

### Human Evaluation

- Customizable evaluation forms
- Multi-criteria assessment
- Inter-annotator agreement calculation
- Qualitative feedback collection

### Evaluation Workflow

1. Generate outputs using different models/prompts
2. Apply automated metrics for initial assessment
3. Conduct human evaluation for qualitative assessment
4. Analyze results and identify improvement areas

### Custom Evaluation Metrics

Custom metrics can be added by extending the `EvaluationMetric` class:

```python
from evaluation.metrics import EvaluationMetric, EvaluationResult

class InsuranceAccuracyMetric(EvaluationMetric):
    def __init__(self):
        super().__init__(
            name="insurance_accuracy",
            description="Measures accuracy of insurance-specific information",
            max_score=1.0
        )
    
    def evaluate(self, generated_text, reference_text, context=None):
        # Implement custom evaluation logic
        score = calculate_accuracy(generated_text, reference_text)
        
        return EvaluationResult(
            metric_name=self.name,
            score=score,
            max_score=self.max_score,
            details={"analysis": "Custom analysis details"}
        )
```

## Benchmarking

The framework includes a comprehensive benchmarking system for comparing model performance:

### Benchmark Datasets

- **Policy Summary**: Benchmark for policy summarization tasks
- **Claim Response**: Benchmark for generating claim responses
- **Customer Inquiry**: Benchmark for handling customer inquiries
- **Risk Assessment**: Benchmark for risk assessment tasks

### Benchmark Structure

Each benchmark consists of:
- Input examples
- Reference outputs
- Evaluation metrics
- Task-specific parameters

### Running Benchmarks

Benchmarks can be run through the UI or programmatically:

```python
from evaluation.benchmarks import BenchmarkManager

# Get benchmark manager
benchmark_manager = BenchmarkManager()

# Run benchmark
results = benchmark_manager.run_benchmark(
    benchmark_name="policy_summary",
    model=model,
    tokenizer=tokenizer,
    inference_engine=inference_engine
)

# Analyze results
average_score = results.get_average_score()
per_example_scores = results.get_per_example_scores()
```

### Comparing Models

The framework provides tools for comparing different models on the same benchmarks:

- Side-by-side output comparison
- Metric comparison charts
- Statistical significance testing
- Performance vs. resource usage analysis

## Extending the Framework

The framework is designed to be extensible in various ways:

### Adding New Models

1. Update `ModelConfig.MODEL_REPOS` with the new model repository
2. Add model details to `ModelConfig.MODEL_DETAILS`
3. Implement any special handling in `ModelLoader.load_model`

### Adding New Prompt Templates

1. Create a new template file in `prompts/templates/`
2. Register the template in the `PromptLibrary`
3. Implement any special handling in `PromptStrategy`

### Adding New Evaluation Metrics

1. Create a new class extending `EvaluationMetric`
2. Implement the `evaluate` method
3. Register the metric in the `MetricsManager`

### Adding New Benchmarks

1. Create benchmark examples in `evaluation/benchmarks/`
2. Implement a benchmark class extending `Benchmark`
3. Register the benchmark in the `BenchmarkManager`

### Adding New UI Components

1. Create a new page class in `ui/pages/`
2. Implement the `render` method
3. Register the page in the main application

## Troubleshooting

### Common Issues and Solutions

#### Model Loading Issues

- **Problem**: "CUDA out of memory" error
  - **Solution**: Use a smaller model or increase quantization (4-bit)

- **Problem**: Model loading is extremely slow on CPU
  - **Solution**: Use a CPU-friendly model like Phi-2 or TinyLLaMA

- **Problem**: "Token not found" error when loading gated models
  - **Solution**: Ensure your HF_TOKEN is correctly set in the .env file

#### Generation Issues

- **Problem**: Generation times out
  - **Solution**: Reduce max_tokens, use a smaller model, or increase timeout

- **Problem**: Poor quality outputs
  - **Solution**: Try a different prompt template or a larger model

- **Problem**: Memory usage grows with each generation
  - **Solution**: Restart the application or use `TorchUtils.clear_gpu_memory()`

#### UI Issues

- **Problem**: Streamlit crashes during model loading
  - **Solution**: Ensure STREAMLIT_WATCHDOG_DISABLE=1 is set

- **Problem**: UI becomes unresponsive during generation
  - **Solution**: Use streaming generation or reduce the generation parameters

### Logging and Debugging

The framework uses Python's logging module for debugging:

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python run.py

# Check log files
cat app.log  # Application logs
cat run.log  # Startup logs
```

## Contributing

Contributions to the Insurance LLM Framework are welcome! Here's how you can contribute:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/insurance-llm-framework.git
cd insurance-llm-framework

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The HuggingFace team for their transformers library
- The Streamlit team for their amazing UI framework
- The open-source LLM community for making powerful models accessible
- All contributors to this project 