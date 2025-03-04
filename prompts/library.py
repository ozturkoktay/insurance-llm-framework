"""
Prompt library for insurance domain tasks.

This module manages a collection of prompt templates for different
insurance-related tasks, such as policy summarization, claim processing,
and customer communication.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
import yaml

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

class PromptTemplate:
    """Class representing a single prompt template."""

    def __init__(
        self,
        name: str,
        template: str,
        task_type: str,
        description: str,
        variables: List[str],
        strategy_type: str = "zero_shot",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a prompt template.

        Args:
            name: Name of the template
            template: The template text with {variable} placeholders
            task_type: The type of task (e.g., policy_summary, claim_response)
            description: Description of the template
            variables: List of variables expected in the template
            strategy_type: Type of prompt strategy to use with this template
            metadata: Additional metadata for the template
        """
        self.name = name
        self.template = template
        self.task_type = task_type
        self.description = description
        self.variables = variables
        self.strategy_type = strategy_type
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization."""
        return {
            "name": self.name,
            "template": self.template,
            "task_type": self.task_type,
            "description": self.description,
            "variables": self.variables,
            "strategy_type": self.strategy_type,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create a template from a dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            task_type=data["task_type"],
            description=data["description"],
            variables=data["variables"],
            strategy_type=data.get("strategy_type", "zero_shot"),
            metadata=data.get("metadata", {})
        )

    def format(self, variables: Dict[str, str]) -> str:
        """
        Format the template with provided variables.

        Args:
            variables: Dictionary of variables to substitute into the template

        Returns:
            Formatted template string
        """
        try:
            return self.template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing template variable: {str(e)}")
            raise ValueError(f"Missing template variable: {str(e)}")

class PromptLibrary:
    """
    Library for managing and accessing prompt templates.

    This class handles loading, saving, and retrieving prompt templates
    for various insurance domain tasks.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the prompt library.

        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self.templates: Dict[str, PromptTemplate] = {}
        self.load_templates()

    def load_templates(self):
        """Load templates from the templates directory."""
        templates_dir = Path(self.templates_dir)
        if not templates_dir.exists():
            logger.warning(
                f"Templates directory not found: {self.templates_dir}")
            templates_dir.mkdir(parents=True, exist_ok=True)
            return

        for file_path in templates_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    template = PromptTemplate.from_dict(data)
                    self.templates[template.name] = template
                    logger.info(f"Loaded template: {template.name}")
            except Exception as e:
                logger.error(
                    f"Error loading template from {file_path}: {str(e)}")

        for file_path in templates_dir.glob("*.yaml"):
            try:
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)
                    template = PromptTemplate.from_dict(data)
                    self.templates[template.name] = template
                    logger.info(f"Loaded template: {template.name}")
            except Exception as e:
                logger.error(
                    f"Error loading template from {file_path}: {str(e)}")

        logger.info(f"Loaded {len(self.templates)} templates")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a template by name.

        Args:
            name: Name of the template to retrieve

        Returns:
            The template or None if not found
        """
        return self.templates.get(name)

    def add_template(self, template: PromptTemplate, save: bool = True):
        """
        Add a template to the library.

        Args:
            template: The template to add
            save: Whether to save the template to disk
        """
        self.templates[template.name] = template

        if save:
            self.save_template(template)

    def save_template(self, template: PromptTemplate):
        """
        Save a template to disk.

        Args:
            template: The template to save
        """
        templates_dir = Path(self.templates_dir)
        templates_dir.mkdir(parents=True, exist_ok=True)

        file_path = templates_dir / f"{template.name}.json"
        try:
            with open(file_path, "w") as f:
                json.dump(template.to_dict(), f, indent=2)
            logger.info(f"Saved template to {file_path}")
        except Exception as e:
            logger.error(f"Error saving template to {file_path}: {str(e)}")

    def remove_template(self, name: str) -> bool:
        """
        Remove a template from the library.

        Args:
            name: Name of the template to remove

        Returns:
            True if successful, False otherwise
        """
        if name not in self.templates:
            return False

        del self.templates[name]

        file_path = Path(self.templates_dir) / f"{name}.json"
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Removed template file: {file_path}")
            except Exception as e:
                logger.error(
                    f"Error removing template file {file_path}: {str(e)}")
                return False

        yaml_path = Path(self.templates_dir) / f"{name}.yaml"
        if yaml_path.exists():
            try:
                yaml_path.unlink()
                logger.info(f"Removed template file: {yaml_path}")
            except Exception as e:
                logger.error(
                    f"Error removing template file {yaml_path}: {str(e)}")
                return False

        return True

    def get_templates_by_task(self, task_type: str) -> List[PromptTemplate]:
        """
        Get all templates for a specific task type.

        Args:
            task_type: Task type to filter by

        Returns:
            List of templates for the specified task
        """
        return [t for t in self.templates.values() if t.task_type == task_type]

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all templates with basic information.

        Returns:
            List of dictionaries with template information
        """
        return [
            {
                "name": t.name,
                "task_type": t.task_type,
                "description": t.description,
                "strategy_type": t.strategy_type
            }
            for t in self.templates.values()
        ]

    def list_task_types(self) -> List[str]:
        """
        List all unique task types in the library.

        Returns:
            List of task types
        """
        return list(set(t.task_type for t in self.templates.values()))

def create_default_templates():
    """Create default insurance prompt templates."""
    templates = [

        PromptTemplate(
            name="policy_summary_concise",
            template=(
                "You are an AI assistant trained to summarize insurance policies clearly and accurately.\n\n"
                "Please provide a concise summary of the following insurance policy. "
                "Focus on key coverage areas, limits, exclusions, and important conditions.\n\n"
                "Policy Document:\n{policy_text}\n\n"
                "Summary:"
            ),
            task_type="policy_summary",
            description="Concise summary of an insurance policy",
            variables=["policy_text"],
            strategy_type="zero_shot"
        ),

        PromptTemplate(
            name="policy_summary_detailed",
            template=(
                "You are an AI assistant trained to summarize insurance policies clearly and accurately.\n\n"
                "Please provide a detailed summary of the following insurance policy. "
                "Include information about:\n"
                "1. Policy type and coverage period\n"
                "2. Coverage areas and limits\n"
                "3. Deductibles and premiums\n"
                "4. Key exclusions and limitations\n"
                "5. Important conditions and responsibilities\n\n"
                "Policy Document:\n{policy_text}\n\n"
                "Detailed Summary:"
            ),
            task_type="policy_summary",
            description="Detailed summary of an insurance policy",
            variables=["policy_text"],
            strategy_type="zero_shot"
        ),

        PromptTemplate(
            name="claim_response_standard",
            template=(
                "You are an AI assistant trained to help insurance professionals draft claim responses.\n\n"
                "Please draft a professional response to the following insurance claim. "
                "Maintain a helpful and empathetic tone while providing clear information.\n\n"
                "Claim Details:\n{claim_text}\n\n"
                "{examples}\n\n"
                "Draft Response:"
            ),
            task_type="claim_response",
            description="Standard response to an insurance claim",
            variables=["claim_text", "examples"],
            strategy_type="few_shot"
        ),

        PromptTemplate(
            name="claim_response_approval",
            template=(
                "You are an AI assistant trained to help insurance professionals draft claim approval responses.\n\n"
                "Please draft a professional, clear, and empathetic response to the following insurance claim that is being approved. The response should:\n\n"
                "1. Clearly state that the claim is approved\n"
                "2. Specify the approved amount and coverage details\n"
                "3. Explain the payment process and timeline\n"
                "4. Provide any necessary next steps for the claimant\n"
                "5. Express appropriate empathy for their situation\n\n"
                "Claim Details:\n{claim_text}\n\n"
                "Approval Information:\n{approval_info}\n\n"
                "Draft Response:"
            ),
            task_type="claim_response",
            description="Professional response for an insurance claim approval",
            variables=["claim_text", "approval_info"],
            strategy_type="zero_shot"
        ),

        PromptTemplate(
            name="claim_response_denial",
            template=(
                "You are an AI assistant trained to help insurance professionals draft claim denial responses.\n\n"
                "Please draft a professional, clear, and empathetic response to the following insurance claim that must be denied. The response should:\n\n"
                "1. Clearly state that the claim is denied\n"
                "2. Explain the specific policy terms or exclusions that apply\n"
                "3. Reference relevant policy sections\n"
                "4. Explain the claimant's options (appeal process, etc.)\n"
                "5. Maintain a respectful and professional tone\n\n"
                "Claim Details:\n{claim_text}\n\n"
                "Policy Information:\n{policy_info}\n\n"
                "Reason for Denial:\n{denial_reason}\n\n"
                "Draft Response:"
            ),
            task_type="claim_response",
            description="Professional response for an insurance claim denial",
            variables=["claim_text", "policy_info", "denial_reason"],
            strategy_type="zero_shot"
        ),

        PromptTemplate(
            name="customer_inquiry_response",
            template=(
                "You are an AI assistant trained to help insurance professionals respond to customer inquiries.\n\n"
                "Please draft a professional and helpful response to the following customer inquiry. "
                "Be clear, concise, and empathetic in your response.\n\n"
                "Customer Inquiry:\n{inquiry_text}\n\n"
                "Draft Response:"
            ),
            task_type="customer_communication",
            description="Response to a customer inquiry",
            variables=["inquiry_text"],
            strategy_type="zero_shot"
        ),

        PromptTemplate(
            name="customer_inquiry_detailed",
            template=(
                "You are an AI assistant trained to help insurance professionals respond to customer inquiries with detailed, accurate information.\n\n"
                "Please draft a comprehensive response to the following customer inquiry about insurance coverage, policies, or claims. The response should:\n\n"
                "1. Address all aspects of the customer's inquiry\n"
                "2. Provide specific, accurate information based on the policy details\n"
                "3. Explain insurance concepts in clear, accessible language\n"
                "4. Offer actionable next steps if applicable\n"
                "5. Maintain a helpful, professional, and empathetic tone\n\n"
                "Customer Inquiry:\n{inquiry_text}\n\n"
                "Relevant Policy Information:\n{policy_details}\n\n"
                "Draft Response:"
            ),
            task_type="customer_communication",
            description="Detailed response to a customer inquiry with specific policy information",
            variables=["inquiry_text", "policy_details"],
            strategy_type="zero_shot"
        ),

        PromptTemplate(
            name="risk_assessment_report",
            template=(
                "You are an AI assistant trained to help insurance professionals with risk assessment.\n\n"
                "Please analyze the following information and create a risk assessment report. "
                "Consider potential risks, their likelihood, impact, and mitigation strategies.\n\n"
                "Risk Information:\n{risk_info}\n\n"
                "{cot_examples}\n\n"
                "{reasoning_request}\n\n"
                "Risk Assessment Report:"
            ),
            task_type="risk_assessment",
            description="Risk assessment report based on provided information",
            variables=["risk_info", "cot_examples", "reasoning_request"],
            strategy_type="chain_of_thought"
        ),

        PromptTemplate(
            name="risk_assessment_detailed",
            template=(
                "You are an AI assistant trained to help insurance professionals with risk assessment.\n\n"
                "Please analyze the following information and create a detailed risk assessment report. Your report should include:\n\n"
                "1. Executive Summary\n"
                "2. Identified Risks (categorized by type)\n"
                "3. Risk Analysis (likelihood and impact assessment)\n"
                "4. Mitigation Strategies\n"
                "5. Recommendations\n\n"
                "Risk Information:\n{risk_info}\n\n"
                "Assessment Scope:\n{assessment_scope}\n\n"
                "Risk Assessment Report:"
            ),
            task_type="risk_assessment",
            description="Create a structured, detailed risk assessment report based on provided information",
            variables=["risk_info", "assessment_scope"],
            strategy_type="zero_shot"
        ),

        PromptTemplate(
            name="policy_comparison",
            template=(
                "You are an AI assistant trained to compare insurance policies clearly and accurately.\n\n"
                "Please provide a detailed comparison of the following insurance policies. Focus on key differences in coverage areas, limits, exclusions, premiums, and important conditions. Highlight advantages and disadvantages of each policy.\n\n"
                "Policy 1:\n{policy_1_text}\n\n"
                "Policy 2:\n{policy_2_text}\n\n"
                "Comparison Analysis:"
            ),
            task_type="policy_comparison",
            description="Compare different insurance policies to highlight key differences, advantages, and disadvantages",
            variables=["policy_1_text", "policy_2_text"],
            strategy_type="zero_shot"
        ),

        PromptTemplate(
            name="compliance_check",
            template=(
                "You are an AI assistant trained to analyze insurance documents for compliance with regulations and standards.\n\n"
                "Please analyze the following document for compliance with insurance regulations and standards. Identify any potential compliance issues, explain why they are problematic, and suggest remediation steps.\n\n"
                "Document to check:\n{document_text}\n\n"
                "Jurisdiction/Regulations to consider:\n{compliance_context}\n\n"
                "Compliance Analysis:"
            ),
            task_type="compliance_check",
            description="Analyze documents or responses for compliance with insurance regulations and standards",
            variables=["document_text", "compliance_context"],
            strategy_type="zero_shot"
        ),

        PromptTemplate(
            name="training_material",
            template=(
                "You are an AI assistant trained to create educational content and training materials for insurance professionals or customers.\n\n"
                "Please create training material on the following insurance topic. The content should be clear, accurate, and tailored to the specified audience. Include relevant examples, key points, and actionable information.\n\n"
                "Topic:\n{topic}\n\n"
                "Target Audience:\n{audience}\n\n"
                "Format:\n{format}\n\n"
                "Training Material:"
            ),
            task_type="training_material",
            description="Create educational content and training materials for insurance professionals or customers",
            variables=["topic", "audience", "format"],
            strategy_type="zero_shot"
        ),
    ]

    templates_dir = Path(DEFAULT_TEMPLATES_DIR)
    templates_dir.mkdir(parents=True, exist_ok=True)

    for template in templates:
        file_path = templates_dir / f"{template.name}.json"
        with open(file_path, "w") as f:
            json.dump(template.to_dict(), f, indent=2)
        logger.info(f"Created default template: {template.name}")

def get_prompt_library():
    """Get or create the global prompt library instance."""

    templates_dir = Path(DEFAULT_TEMPLATES_DIR)
    if not templates_dir.exists() or not any(templates_dir.glob("*.json")):
        create_default_templates()

    return PromptLibrary()
