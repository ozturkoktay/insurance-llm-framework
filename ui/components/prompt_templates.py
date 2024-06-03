"""
Prompt Templates Component for the Insurance LLM Framework.

This module provides UI components for managing prompt templates.
"""

import streamlit as st
import os
import json
import logging
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Default template directory
TEMPLATE_DIR = os.path.join(os.getcwd(), "templates")


def get_template_directory() -> str:
    """Get the directory where templates are stored."""
    # Create template directory if it doesn't exist
    if not os.path.exists(TEMPLATE_DIR):
        os.makedirs(TEMPLATE_DIR)
        logger.info(f"Created template directory at {TEMPLATE_DIR}")

        # Create sample templates
        create_sample_templates()

    return TEMPLATE_DIR


def create_sample_templates() -> None:
    """Create sample templates for the framework."""
    sample_templates = [
        {
            "id": "policy_summarization_zero_shot",
            "name": "Policy Summarization (Zero-shot)",
            "description": "Summarize an insurance policy document in simple terms",
            "format": "zero-shot",
            "task_type": "summarization",
            "insurance_domain": "general",
            "template": """You are an expert insurance policy analyst. 
Your task is to summarize the following insurance policy document in simple, clear language that the average consumer could understand.

Focus on:
- Main coverage provided
- Key exclusions and limitations
- Important conditions or requirements
- Deductibles and limits
- Any unusual or notable features

Provide a structured summary with appropriate headings.

POLICY DOCUMENT:
{policy_document}

SUMMARY:""",
            "input_variables": ["policy_document"],
            "created_at": "2023-10-01T12:00:00Z",
            "updated_at": "2023-10-01T12:00:00Z",
            "tags": ["policy", "summarization", "zero-shot"]
        },
        {
            "id": "claim_response_few_shot",
            "name": "Claim Response (Few-shot)",
            "description": "Generate a response to an insurance claim based on policy coverage",
            "format": "few-shot",
            "task_type": "generation",
            "insurance_domain": "claims",
            "template": """You are an insurance claims specialist. 
Write a professional and empathetic response to the following claim, explaining the coverage decision based on the policy details.

Here are some examples of good claim responses:

EXAMPLE 1:
CLAIM: My basement flooded during a heavy rainstorm, causing damage to my furniture and carpeting. I have a homeowner's policy and would like to claim for water damage.
POLICY COVERAGE: The policy covers water damage from plumbing failures but excludes flooding from external sources including rainwater, overflowing bodies of water, and ground seepage.
RESPONSE: 
Dear Policyholder,

Thank you for submitting your claim regarding water damage to your basement. After careful review of your policy, I regret to inform you that we are unable to cover this specific damage. Your homeowner's policy explicitly excludes water damage resulting from external flooding, rainwater, or ground seepage.

While your policy does provide coverage for water damage, this is limited to incidents involving internal plumbing failures such as burst pipes or appliance leaks. The damage described in your claim resulted from external rainwater, which falls under the flooding exclusion in your policy.

I understand this is disappointing news. You might want to check if you have a separate flood insurance policy that could cover this incident. If you don't currently have flood coverage, I'd be happy to provide information about our flood insurance options to protect you from similar situations in the future.

Please don't hesitate to contact me if you have any questions about your coverage or if you'd like to discuss flood insurance options.

Sincerely,
Claims Specialist

EXAMPLE 2:
CLAIM: My car was hit while parked in a shopping mall parking lot. The other driver left a note with their insurance information.
POLICY COVERAGE: Comprehensive collision coverage with $500 deductible. Policy includes coverage for repairs at any certified auto body shop.
RESPONSE:
Dear Policyholder,

I'm pleased to inform you that your claim for the parking lot damage to your vehicle has been approved. Your policy's collision coverage applies to this incident, and we will proceed with covering the cost of repairs minus your $500 deductible.

You may take your vehicle to any certified auto body shop of your choice for repairs. Once you've selected a repair facility, please provide us with their information, and we will work directly with them regarding payment.

Additionally, since the other driver left their insurance information, we will initiate the subrogation process with their insurance company. If we successfully recover the costs from them, we will reimburse your $500 deductible.

Please forward any repair estimates you receive to our claims department. If you need a rental car during repairs, your policy includes rental coverage for up to $30 per day for a maximum of 14 days.

If you have any questions or need assistance with the repair process, please don't hesitate to contact me.

Sincerely,
Claims Specialist

Now, please respond to the following claim:

CLAIM: {claim_description}
POLICY COVERAGE: {policy_coverage}
RESPONSE:""",
            "input_variables": ["claim_description", "policy_coverage"],
            "created_at": "2023-10-05T14:30:00Z",
            "updated_at": "2023-10-05T14:30:00Z",
            "tags": ["claims", "response", "few-shot"]
        },
        {
            "id": "risk_assessment_cot",
            "name": "Risk Assessment (Chain-of-Thought)",
            "description": "Analyze insurance risk factors using chain-of-thought reasoning",
            "format": "chain-of-thought",
            "task_type": "analysis",
            "insurance_domain": "underwriting",
            "template": """You are an experienced insurance risk analyst. 
Assess the risk factors in the following scenario and determine an appropriate risk classification.

Think through this step-by-step:
1. Identify all potential risk factors in the scenario
2. Evaluate each risk factor's severity and likelihood
3. Consider mitigating factors that may reduce risk
4. Determine how these factors interact with each other
5. Classify the overall risk as Low, Medium, High, or Extreme
6. Recommend conditions or exclusions that should apply

SCENARIO:
{risk_scenario}

THOUGHT PROCESS:""",
            "input_variables": ["risk_scenario"],
            "created_at": "2023-11-10T09:15:00Z",
            "updated_at": "2023-11-10T09:15:00Z",
            "tags": ["underwriting", "risk", "chain-of-thought"]
        },
        {
            "id": "customer_inquiry_response",
            "name": "Customer Inquiry Response",
            "description": "Generate responses to customer inquiries about insurance policies",
            "format": "zero-shot",
            "task_type": "generation",
            "insurance_domain": "customer service",
            "template": """You are a knowledgeable and helpful insurance customer service representative.
Your goal is to provide clear, accurate, and helpful responses to customer inquiries about their insurance policies.

Guidelines:
- Be professional but conversational in tone
- Provide specific and accurate information based on the policy details
- Avoid insurance jargon when possible, or explain it when used
- Address all parts of the customer's inquiry
- If you can't provide specific information, explain what the customer should do next

POLICY INFORMATION:
{policy_details}

CUSTOMER INQUIRY:
{customer_question}

RESPONSE:""",
            "input_variables": ["policy_details", "customer_question"],
            "created_at": "2023-12-05T10:45:00Z",
            "updated_at": "2023-12-05T10:45:00Z",
            "tags": ["customer service", "inquiry", "response"]
        }
    ]

    # Save sample templates
    for template in sample_templates:
        template_path = os.path.join(TEMPLATE_DIR, f"{template['id']}.json")
        with open(template_path, "w") as f:
            json.dump(template, f, indent=2)

        logger.info(f"Created sample template: {template['name']}")


def load_templates() -> List[Dict[str, Any]]:
    """
    Load all templates from the template directory.

    Returns:
        List of template dictionaries
    """
    template_dir = get_template_directory()
    templates = []

    # List all JSON files in template directory
    template_files = [f for f in os.listdir(
        template_dir) if f.endswith(".json")]

    for file in template_files:
        try:
            with open(os.path.join(template_dir, file), "r") as f:
                template = json.load(f)
                templates.append(template)
        except Exception as e:
            logger.error(f"Error loading template {file}: {str(e)}")

    # Sort templates by name
    templates.sort(key=lambda x: x.get("name", ""))

    return templates


def save_template(template: Dict[str, Any]) -> bool:
    """
    Save a template to the template directory.

    Args:
        template: Template dictionary to save

    Returns:
        True if successful, False otherwise
    """
    try:
        template_dir = get_template_directory()

        # Ensure template has an ID
        if "id" not in template:
            template["id"] = str(uuid.uuid4())

        # Add timestamps
        if "created_at" not in template:
            template["created_at"] = datetime.now().strftime(
                "%Y-%m-%dT%H:%M:%SZ")

        template["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Save template to file
        template_path = os.path.join(template_dir, f"{template['id']}.json")
        with open(template_path, "w") as f:
            json.dump(template, f, indent=2)

        logger.info(f"Saved template: {template['name']}")
        return True
    except Exception as e:
        logger.error(f"Error saving template: {str(e)}")
        return False


def delete_template(template_id: str) -> bool:
    """
    Delete a template from the template directory.

    Args:
        template_id: ID of the template to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        template_dir = get_template_directory()
        template_path = os.path.join(template_dir, f"{template_id}.json")

        if os.path.exists(template_path):
            os.remove(template_path)
            logger.info(f"Deleted template: {template_id}")
            return True
        else:
            logger.warning(f"Template not found: {template_id}")
            return False
    except Exception as e:
        logger.error(f"Error deleting template: {str(e)}")
        return False


def template_selector() -> Optional[Dict[str, Any]]:
    """
    Display a UI for selecting a prompt template.

    Returns:
        Selected template dictionary, or None if no template is selected
    """
    # Load all templates
    templates = load_templates()

    if not templates:
        st.info("No templates found. Create a new template to get started.")
        return None

    # Create filter options
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        task_types = [
            "All"] + sorted(list(set(t.get("task_type", "Other") for t in templates)))
        selected_task = st.selectbox("Filter by Task Type", options=task_types)

    with col2:
        formats = ["All"] + \
            sorted(list(set(t.get("format", "Other") for t in templates)))
        selected_format = st.selectbox("Filter by Format", options=formats)

    with col3:
        domains = [
            "All"] + sorted(list(set(t.get("insurance_domain", "Other") for t in templates)))
        selected_domain = st.selectbox("Filter by Domain", options=domains)

    # Apply filters
    filtered_templates = templates
    if selected_task != "All":
        filtered_templates = [t for t in filtered_templates if t.get(
            "task_type") == selected_task]
    if selected_format != "All":
        filtered_templates = [t for t in filtered_templates if t.get(
            "format") == selected_format]
    if selected_domain != "All":
        filtered_templates = [t for t in filtered_templates if t.get(
            "insurance_domain") == selected_domain]

    if not filtered_templates:
        st.warning("No templates match the selected filters.")
        return None

    # Create a table of templates
    template_data = [
        {
            "Name": t.get("name", "Unnamed Template"),
            "Description": t.get("description", "No description"),
            "Format": t.get("format", "").capitalize(),
            "Task Type": t.get("task_type", "").capitalize(),
            "Domain": t.get("insurance_domain", "").capitalize(),
        }
        for t in filtered_templates
    ]

    template_df = pd.DataFrame(template_data)
    st.dataframe(template_df, use_container_width=True, hide_index=True)

    # Template selection
    template_names = [t.get("name", f"Template {i}")
                      for i, t in enumerate(filtered_templates)]
    selected_name = st.selectbox("Select a Template", options=template_names)

    # Find the selected template
    selected_template = next(
        (t for t in filtered_templates if t.get("name") == selected_name), None)

    return selected_template


def template_preview(template: Dict[str, Any]) -> None:
    """
    Display a preview of a prompt template.

    Args:
        template: Template dictionary to preview
    """
    st.markdown(f"### {template.get('name', 'Unnamed Template')}")
    st.markdown(template.get("description", "No description provided."))

    # Display template metadata
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(
            f"**Format:** {template.get('format', 'Not specified').capitalize()}")

    with col2:
        st.markdown(
            f"**Task Type:** {template.get('task_type', 'Not specified').capitalize()}")

    with col3:
        st.markdown(
            f"**Domain:** {template.get('insurance_domain', 'Not specified').capitalize()}")

    # Display template content
    st.markdown("#### Template Content")
    st.code(template.get(
        "template", "Template content not available."), language="text")

    # Display input variables
    if "input_variables" in template and template["input_variables"]:
        st.markdown("#### Input Variables")
        for var in template["input_variables"]:
            st.markdown(f"- `{var}`")

    # Display tags if available
    if "tags" in template and template["tags"]:
        st.markdown("#### Tags")
        tags_html = " ".join(
            [f"<span style='background-color: #f0f2f6; padding: 0.2rem 0.5rem; border-radius: 0.5rem; margin-right: 0.5rem;'>{tag}</span>" for tag in template["tags"]])
        st.markdown(tags_html, unsafe_allow_html=True)

    # Try button
    if st.button("Try this Template", key="try_template"):
        # Set this template as active
        st.session_state.active_template = template

        # Provide feedback
        st.success(
            f"Template '{template.get('name')}' is now active. Go to the Text Generation page to use it.")


def template_editor(template: Optional[Dict[str, Any]] = None) -> None:
    """
    Display an editor for creating or modifying prompt templates.

    Args:
        template: Optional template to edit (None for new template)
    """
    # Create a form for the template editor
    with st.form("template_editor_form"):
        # Template identification
        template_name = st.text_input(
            "Template Name",
            value=template.get("name", "") if template else "",
            help="A short, descriptive name for the template"
        )

        template_description = st.text_area(
            "Description",
            value=template.get("description", "") if template else "",
            help="A brief description of what this template does"
        )

        # Template metadata
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            format_options = ["zero-shot", "few-shot",
                              "chain-of-thought", "self-consistency", "other"]
            template_format = st.selectbox(
                "Format",
                options=format_options,
                index=format_options.index(template.get(
                    "format", "zero-shot")) if template and template.get("format") in format_options else 0,
                help="The prompt engineering strategy used"
            )

        with col2:
            task_options = ["summarization", "generation",
                            "analysis", "classification", "extraction", "other"]
            task_type = st.selectbox(
                "Task Type",
                options=task_options,
                index=task_options.index(template.get("task_type", "generation")) if template and template.get(
                    "task_type") in task_options else 1,
                help="The type of task this template is designed for"
            )

        with col3:
            domain_options = ["general", "claims", "underwriting",
                              "policy", "customer service", "compliance", "other"]
            insurance_domain = st.selectbox(
                "Insurance Domain",
                options=domain_options,
                index=domain_options.index(template.get("insurance_domain", "general")) if template and template.get(
                    "insurance_domain") in domain_options else 0,
                help="The insurance domain this template is designed for"
            )

        # Template content
        template_content = st.text_area(
            "Template Content",
            value=template.get("template", "") if template else "",
            height=300,
            help="The actual template text. Use {variable_name} for placeholders."
        )

        # Extract input variables from template content
        input_vars = []
        if template_content:
            import re
            input_vars = list(set(re.findall(r"\{(\w+)\}", template_content)))

        # Display and edit input variables
        st.markdown("#### Input Variables")
        st.markdown(
            "Variables automatically detected from template content. Add or remove as needed.")

        # Show input variables from the template if available
        existing_vars = template.get("input_variables", []) if template else []
        all_vars = sorted(list(set(existing_vars + input_vars)))

        # Allow editing input variables
        input_variables = st.multiselect(
            "Input Variables",
            options=all_vars,
            default=all_vars,
            help="Variables that need to be filled in when using this template"
        )

        # Tags for the template
        tags_input = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(template.get("tags", [])
                            ) if template and "tags" in template else "",
            help="Tags to help categorize and find this template"
        )

        # Process tags
        tags = [tag.strip()
                for tag in tags_input.split(",")] if tags_input else []
        tags = [tag for tag in tags if tag]  # Remove empty tags

        # Submit button
        submit_label = "Update Template" if template else "Create Template"
        submitted = st.form_submit_button(
            submit_label, use_container_width=True)

        if submitted:
            if not template_name:
                st.error("Template name is required")
                return

            if not template_content:
                st.error("Template content is required")
                return

            # Create template dictionary
            new_template = {
                "id": template.get("id", str(uuid.uuid4())) if template else str(uuid.uuid4()),
                "name": template_name,
                "description": template_description,
                "format": template_format,
                "task_type": task_type,
                "insurance_domain": insurance_domain,
                "template": template_content,
                "input_variables": input_variables,
                "tags": tags
            }

            # Add timestamps
            if template and "created_at" in template:
                new_template["created_at"] = template["created_at"]

            # Save the template
            if save_template(new_template):
                st.success(f"Template '{template_name}' saved successfully")

                # Clear editing state
                if "editing_template" in st.session_state:
                    st.session_state.editing_template = None

                # Optionally set as active template
                if st.checkbox("Set as active template", value=True):
                    st.session_state.active_template = new_template
                    st.success(f"Template '{template_name}' set as active")

                # Force UI refresh
                st.rerun()
            else:
                st.error("Error saving template")

    # Delete button (outside the form, only for existing templates)
    if template and "id" in template:
        if st.button("Delete Template", type="secondary"):
            if delete_template(template["id"]):
                st.success(
                    f"Template '{template.get('name', 'Unnamed')}' deleted successfully")

                # Clear editing state
                if "editing_template" in st.session_state:
                    st.session_state.editing_template = None

                # Clear active template if it's the same one
                if "active_template" in st.session_state and st.session_state.active_template and st.session_state.active_template.get("id") == template["id"]:
                    st.session_state.active_template = None

                # Force UI refresh
                st.rerun()
            else:
                st.error("Error deleting template")


def display_template_info(template: Dict[str, Any]) -> None:
    """
    Display detailed information about a template.

    Args:
        template: Template dictionary to display information for
    """
    # Basic template information
    st.markdown(f"### {template.get('name', 'Unnamed Template')}")
    st.markdown(template.get("description", "No description provided."))

    # Template metadata
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.metric("Format", template.get(
            "format", "Not specified").capitalize())

    with col2:
        st.metric("Task Type", template.get(
            "task_type", "Not specified").capitalize())

    with col3:
        st.metric("Domain", template.get(
            "insurance_domain", "Not specified").capitalize())

    # Created and updated timestamps
    if "created_at" in template or "updated_at" in template:
        created = template.get("created_at", "Unknown")
        updated = template.get("updated_at", "Unknown")

        # Format dates if they are in ISO format
        try:
            created_dt = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
            created = created_dt.strftime("%b %d, %Y")
        except (ValueError, TypeError):
            pass

        try:
            updated_dt = datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ")
            updated = updated_dt.strftime("%b %d, %Y")
        except (ValueError, TypeError):
            pass

        col1, col2 = st.columns([1, 1])
        with col1:
            st.caption(f"Created: {created}")
        with col2:
            st.caption(f"Last Updated: {updated}")

    # Template content in expandable section
    with st.expander("Template Content", expanded=True):
        st.code(template.get(
            "template", "Template content not available."), language="text")

    # Display template examples
    with st.expander("Example Usage", expanded=False):
        st.markdown("#### Example Prompt")

        # Create example values for input variables
        example_values = {}
        for var in template.get("input_variables", []):
            if var == "policy_document":
                example_values[var] = "This homeowner's insurance policy provides coverage for your dwelling, personal property, and liability..."
            elif var == "claim_description":
                example_values[var] = "My car was damaged in a hailstorm last Tuesday while parked in my driveway."
            elif var == "policy_coverage":
                example_values[var] = "Comprehensive coverage with $500 deductible. Weather-related damage is covered."
            elif var == "risk_scenario":
                example_values[var] = "A 30-year-old commercial building in a flood zone with outdated electrical systems."
            elif var == "policy_details":
                example_values[var] = "Auto insurance policy #123456, comprehensive coverage, $1,000 deductible, roadside assistance included."
            elif var == "customer_question":
                example_values[var] = "Does my policy cover rental cars if my vehicle is being repaired after an accident?"
            else:
                example_values[var] = f"[Example {var} content]"

        # Fill in the template with example values
        example_prompt = template.get("template", "")
        for var, value in example_values.items():
            example_prompt = example_prompt.replace(f"{{{var}}}", value)

        st.code(example_prompt, language="text")

    # Display usage suggestions
    with st.expander("Usage Suggestions", expanded=False):
        st.markdown("#### Recommended Use Cases")

        # Determine suggestions based on template metadata
        task_type = template.get("task_type", "").lower()
        format_type = template.get("format", "").lower()
        domain = template.get("insurance_domain", "").lower()

        suggestions = []

        if task_type == "summarization":
            suggestions.append(
                "- Condensing long policy documents into customer-friendly summaries")
            suggestions.append(
                "- Creating executive summaries of claim reports")

        if task_type == "generation":
            suggestions.append("- Creating claim response letters")
            suggestions.append(
                "- Drafting policy explanations for specific scenarios")

        if task_type == "analysis":
            suggestions.append(
                "- Identifying risk factors in property descriptions")
            suggestions.append(
                "- Analyzing claim patterns for fraud detection")

        if task_type == "classification":
            suggestions.append("- Categorizing claims by type and severity")
            suggestions.append("- Sorting customer inquiries by department")

        if domain == "claims":
            suggestions.append(
                "- Processing and responding to claim submissions")
            suggestions.append("- Explaining claim decisions to policyholders")

        if domain == "underwriting":
            suggestions.append(
                "- Evaluating risk factors for new policy applications")
            suggestions.append("- Identifying potential coverage issues")

        if domain == "customer service":
            suggestions.append("- Answering common insurance questions")
            suggestions.append("- Explaining policy details in simple terms")

        if not suggestions:
            suggestions.append("- General insurance document processing")
            suggestions.append(
                "- Custom prompt engineering for specific needs")

        for suggestion in suggestions:
            st.markdown(suggestion)

        st.markdown("#### Parameter Recommendations")

        # Suggest parameters based on task and format
        if "chain-of-thought" in format_type:
            st.markdown(
                "- Use higher max_tokens (1500+) to allow for detailed reasoning")
            st.markdown(
                "- Set lower temperature (0.1-0.3) for more consistent analysis")
        elif "few-shot" in format_type:
            st.markdown(
                "- Balance temperature (0.5-0.7) for creativity while following examples")
            st.markdown(
                "- Consider presence_penalty (0.1-0.5) to reduce repetition")
        elif task_type == "generation":
            st.markdown(
                "- Moderate temperature (0.6-0.8) for natural-sounding text")
            st.markdown("- Adjust max_tokens based on desired response length")
        elif task_type == "summarization":
            st.markdown(
                "- Lower temperature (0.3-0.5) for more factual summaries")
            st.markdown(
                "- Set appropriate max_tokens to constrain summary length")


def fill_template(template: Dict[str, Any], values: Dict[str, str]) -> str:
    """
    Fill a template with values.

    Args:
        template: Template dictionary
        values: Dictionary of values to fill in

    Returns:
        Filled template string
    """
    template_text = template.get("template", "")

    # Replace variables with values
    for var, value in values.items():
        template_text = template_text.replace(f"{{{var}}}", value)

    return template_text


if __name__ == "__main__":
    # For testing the component in isolation
    st.set_page_config(
        page_title="Prompt Templates - Insurance LLM Framework",
        page_icon="📝",
        layout="wide"
    )

    st.title("Prompt Templates Component Test")

    # Initialize session state for testing
    if "active_template" not in st.session_state:
        st.session_state.active_template = None

    # Test the component
    st.subheader("Template Selector")
    selected_template = template_selector()

    if selected_template:
        st.success(f"Selected template: {selected_template.get('name')}")

        # Test template preview
        st.subheader("Template Preview")
        template_preview(selected_template)

        # Test template editor
        st.subheader("Template Editor")
        template_editor(selected_template)
    else:
        # Test creating a new template
        st.subheader("Create New Template")
        template_editor()
