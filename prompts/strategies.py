"""
Prompt engineering strategies for insurance domain tasks.

This module provides various prompt engineering approaches for different
insurance tasks, such as zero-shot, few-shot, and chain-of-thought prompting.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PromptStrategy:
    """Base class for prompt engineering strategies."""

    def __init__(self, name: str, description: str):
        """
        Initialize a prompt strategy.

        Args:
            name: Name of the strategy
            description: Description of the strategy
        """
        self.name = name
        self.description = description

    def format_prompt(self, template: str, variables: Dict[str, str]) -> str:
        """
        Format a prompt template with variables.

        Args:
            template: Prompt template with {variable} placeholders
            variables: Dictionary of variables to substitute

        Returns:
            Formatted prompt
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing prompt variable: {str(e)}")
            raise ValueError(f"Missing template variable: {str(e)}")

    def apply(self, task_input: Dict[str, Any], template: str) -> str:
        """
        Apply the prompt strategy to format an input for a model.

        Args:
            task_input: Input data for the task
            template: Base template to use

        Returns:
            Formatted prompt
        """
        raise NotImplementedError("Subclasses must implement this method")


class ZeroShotPromptStrategy(PromptStrategy):
    """Zero-shot prompting strategy for insurance tasks."""

    def __init__(self):
        """Initialize the zero-shot prompting strategy."""
        super().__init__(
            name="zero_shot",
            description="Direct instruction without examples"
        )

    def apply(self, task_input: Dict[str, Any], template: str) -> str:
        """
        Apply zero-shot prompting strategy.

        Args:
            task_input: Input data for the task
            template: Base template to use

        Returns:
            Formatted prompt for zero-shot inference
        """
        return self.format_prompt(template, task_input)


class FewShotPromptStrategy(PromptStrategy):
    """Few-shot prompting strategy for insurance tasks."""

    def __init__(self, examples: List[Dict[str, Any]], example_template: str):
        """
        Initialize the few-shot prompting strategy.

        Args:
            examples: List of example inputs and outputs
            example_template: Template for formatting individual examples
        """
        super().__init__(
            name="few_shot",
            description="Instruction with examples"
        )
        self.examples = examples
        self.example_template = example_template

    def apply(self, task_input: Dict[str, Any], template: str) -> str:
        """
        Apply few-shot prompting strategy.

        Args:
            task_input: Input data for the task
            template: Base template to use

        Returns:
            Formatted prompt with examples followed by the task
        """
        # Format each example using the example template
        formatted_examples = []
        for example in self.examples:
            formatted_example = self.format_prompt(
                self.example_template, example)
            formatted_examples.append(formatted_example)

        # Combine examples into a single string
        examples_text = "\n\n".join(formatted_examples)

        # Add examples to the task_input
        task_input_with_examples = task_input.copy()
        task_input_with_examples["examples"] = examples_text

        # Format the final prompt
        return self.format_prompt(template, task_input_with_examples)


class ChainOfThoughtPromptStrategy(PromptStrategy):
    """Chain-of-thought prompting strategy for complex insurance reasoning tasks."""

    def __init__(self, cot_examples: List[Dict[str, Any]], cot_example_template: str):
        """
        Initialize the chain-of-thought prompting strategy.

        Args:
            cot_examples: List of examples with reasoning steps
            cot_example_template: Template for formatting individual examples with reasoning
        """
        super().__init__(
            name="chain_of_thought",
            description="Instruction with step-by-step reasoning examples"
        )
        self.cot_examples = cot_examples
        self.cot_example_template = cot_example_template

    def apply(self, task_input: Dict[str, Any], template: str) -> str:
        """
        Apply chain-of-thought prompting strategy.

        Args:
            task_input: Input data for the task
            template: Base template to use

        Returns:
            Formatted prompt with reasoning examples followed by the task
        """
        # Format each example using the example template
        formatted_examples = []
        for example in self.cot_examples:
            formatted_example = self.format_prompt(
                self.cot_example_template, example)
            formatted_examples.append(formatted_example)

        # Combine examples into a single string
        examples_text = "\n\n".join(formatted_examples)

        # Add examples to the task_input
        task_input_with_examples = task_input.copy()
        task_input_with_examples["cot_examples"] = examples_text

        # Format the final prompt with explicit request for reasoning
        task_input_with_examples["reasoning_request"] = "Let's work through this step by step:"

        return self.format_prompt(template, task_input_with_examples)


class ReActPromptStrategy(PromptStrategy):
    """ReAct (Reasoning + Acting) prompting strategy for complex insurance tasks."""

    def __init__(self, react_examples: List[Dict[str, Any]], react_example_template: str):
        """
        Initialize the ReAct prompting strategy.

        Args:
            react_examples: List of examples with reasoning and actions
            react_example_template: Template for formatting examples with reasoning and actions
        """
        super().__init__(
            name="react",
            description="Reasoning and acting prompting for complex tasks"
        )
        self.react_examples = react_examples
        self.react_example_template = react_example_template

    def apply(self, task_input: Dict[str, Any], template: str) -> str:
        """
        Apply ReAct prompting strategy.

        Args:
            task_input: Input data for the task
            template: Base template to use

        Returns:
            Formatted prompt with reasoning and acting examples
        """
        # Format each example using the example template
        formatted_examples = []
        for example in self.react_examples:
            formatted_example = self.format_prompt(
                self.react_example_template, example)
            formatted_examples.append(formatted_example)

        # Combine examples into a single string
        examples_text = "\n\n".join(formatted_examples)

        # Add examples to the task_input
        task_input_with_examples = task_input.copy()
        task_input_with_examples["react_examples"] = examples_text

        # Format the final prompt
        return self.format_prompt(template, task_input_with_examples)


# Factory function to create prompt strategies
def create_prompt_strategy(
    strategy_type: str,
    **kwargs
) -> PromptStrategy:
    """
    Create a prompt strategy of the specified type.

    Args:
        strategy_type: Type of prompt strategy to create
        **kwargs: Additional arguments for the specific strategy

    Returns:
        An instance of the specified PromptStrategy
    """
    if strategy_type == "zero_shot":
        return ZeroShotPromptStrategy()
    elif strategy_type == "few_shot":
        examples = kwargs.get("examples", [])
        example_template = kwargs.get("example_template", "")
        return FewShotPromptStrategy(examples, example_template)
    elif strategy_type == "chain_of_thought":
        cot_examples = kwargs.get("cot_examples", [])
        cot_example_template = kwargs.get("cot_example_template", "")
        return ChainOfThoughtPromptStrategy(cot_examples, cot_example_template)
    elif strategy_type == "react":
        react_examples = kwargs.get("react_examples", [])
        react_example_template = kwargs.get("react_example_template", "")
        return ReActPromptStrategy(react_examples, react_example_template)
    else:
        raise ValueError(f"Unknown prompt strategy type: {strategy_type}")
