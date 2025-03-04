"""
Text Generation Page for the Insurance LLM Framework.

This module provides the interface for generating text using selected models and templates.
"""

import streamlit as st
import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd

from ui.components.file_uploader import file_uploader, display_document_library
from ui.components.prompt_templates import fill_template
from ui.components.output_display import display_generated_text

logger = logging.getLogger(__name__)

def render():
    """Render the text generation page."""
    st.title("🔄 Text Generation")

    with st.expander("ℹ️ About Text Generation", expanded=False):
        st.markdown("""
        This page allows you to generate text for insurance tasks using the selected model and prompt template.

        1. Select an insurance document, use a sample document, or enter text manually
        2. Review and modify the filled prompt template if needed
        3. Configure generation parameters (if desired)
        4. Generate text using the selected model
        5. View, evaluate, and export the generated text
        
        For best results, make sure you have selected an appropriate model and prompt template.
        """)

    if "active_model" not in st.session_state or not st.session_state.active_model:
        st.warning(
            "⚠️ No model is currently loaded. Please select a model in the Model Selection page.")
        if st.button("Go to Model Selection"):

            st.info("In a real app, this would navigate to the Model Selection page.")
        return

    if "active_template" not in st.session_state or not st.session_state.active_template:
        st.warning(
            "⚠️ No prompt template is selected. Please select a template in the Prompt Engineering page.")
        if st.button("Go to Prompt Engineering"):

            st.info(
                "In a real app, this would navigate to the Prompt Engineering page.")
        return

    template = st.session_state.active_template
    model_name = st.session_state.active_model

    with st.sidebar:
        st.markdown("### Generation Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.model_config.get(
                "temperature", 0.7) if "model_config" in st.session_state else 0.7,
            step=0.1,
            help="Higher values produce more diverse outputs"
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=4000,
            value=st.session_state.model_config.get(
                "max_tokens", 1000) if "model_config" in st.session_state else 1000,
            step=50,
            help="Maximum number of tokens to generate"
        )

        st.session_state.generation_params = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    tabs = st.tabs(["Document Input", "Prompt", "Generation"])

    with tabs[0]:
        st.subheader("Insurance Document Input")

        file_type = "text"
        task_type = template.get("task_type", "").lower()

        if "policy" in template.get("tags", []) or "policy" in template.get("name", "").lower():
            file_type = "policy"
        elif "claim" in template.get("tags", []) or "claim" in template.get("name", "").lower():
            file_type = "claim"
        elif "communication" in template.get("tags", []) or "customer" in template.get("name", "").lower():
            file_type = "communication"

        uploaded_content = file_uploader(
            file_type=file_type,
            allowed_types=["txt", "pdf", "docx"],
            default_dir=f"data/{file_type}_samples" if file_type in [
                "policy", "claim", "communication"] else "data/samples"
        )

        if uploaded_content:
            st.session_state.document_content = uploaded_content
            st.success("Document content loaded successfully.")

        st.subheader("Document Library")
        display_document_library(default_dir="data/saved_documents")

    with tabs[1]:
        st.subheader("Prompt Template")

        st.markdown(
            f"**Active Template:** {template.get('name', 'Unnamed Template')}")
        st.markdown(f"*{template.get('description', 'No description')}*")

        st.markdown("### Fill Template Variables")

        input_variables = template.get("input_variables", [])

        if "template_inputs" not in st.session_state:
            st.session_state.template_inputs = {}

        if "document_content" in st.session_state:

            if "policy_document" in input_variables:
                st.session_state.template_inputs["policy_document"] = st.session_state.document_content
            elif "claim_description" in input_variables:
                st.session_state.template_inputs["claim_description"] = st.session_state.document_content
            elif "risk_scenario" in input_variables:
                st.session_state.template_inputs["risk_scenario"] = st.session_state.document_content
            elif "customer_question" in input_variables:
                st.session_state.template_inputs["customer_question"] = st.session_state.document_content
            elif len(input_variables) > 0:

                st.session_state.template_inputs[input_variables[0]
                                                 ] = st.session_state.document_content

        with st.form("template_variables_form"):
            for var in input_variables:

                default_value = st.session_state.template_inputs.get(var, "")

                if var in ["policy_document", "claim_description", "risk_scenario"]:
                    var_value = st.text_area(
                        f"{var.replace('_', ' ').title()}",
                        value=default_value,
                        height=200
                    )
                else:
                    var_value = st.text_area(
                        f"{var.replace('_', ' ').title()}",
                        value=default_value,
                        height=100
                    )

                st.session_state.template_inputs[var] = var_value

            submitted = st.form_submit_button(
                "Fill Template", use_container_width=True)

        st.markdown("### Filled Prompt")

        filled_template = fill_template(
            template, st.session_state.template_inputs)

        edited_prompt = st.text_area(
            "Edit Prompt (if needed)",
            value=filled_template,
            height=300
        )

        st.session_state.current_prompt = edited_prompt

        if st.button("Save Prompt", use_container_width=True):
            st.session_state.current_prompt = edited_prompt
            st.success("Prompt saved for generation.")

    with tabs[2]:
        st.subheader("Text Generation")

        st.info(
            f"Model: **{model_name}** | Template: **{template.get('name', 'Unnamed')}**")

        if "current_prompt" not in st.session_state or not st.session_state.current_prompt:
            st.warning(
                "Please fill in the template variables and save the prompt first.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Temperature", f"{temperature}",
                      help="Controls randomness of generation")
        with col2:
            st.metric("Max Tokens", f"{max_tokens}",
                      help="Controls maximum length of generated text")

        if st.button("Generate Text", type="primary", use_container_width=True):
            with st.spinner("Generating text..."):

                output_container = st.empty()
                generated_text = ""

                task_type = template.get("task_type", "").lower()

                if task_type == "summarization":
                    full_text = "This insurance policy provides coverage for property damage and liability. Key coverages include dwelling protection up to $300,000, personal property coverage up to $150,000, and liability coverage up to $500,000. Notable exclusions include flood damage, intentional acts, and wear and tear. The policy requires a $1,000 deductible for most claims."
                elif task_type == "analysis":
                    full_text = "Risk Assessment Analysis:\n\n1. Primary Risk Factors:\n   - Commercial building age (30 years) indicates potential structural and system deterioration\n   - Location in flood zone presents high water damage risk\n   - Outdated electrical systems pose significant fire hazard\n\n2. Risk Severity and Likelihood:\n   - Flood risk: High severity, moderate-to-high likelihood based on location\n   - Electrical system fire: High severity, moderate likelihood\n   - Structural issues: Moderate severity, moderate likelihood\n\n3. Mitigating Factors:\n   - None mentioned in scenario\n   - Would need to assess existing flood prevention measures\n   - Would need to verify building code compliance status\n\n4. Risk Interaction Analysis:\n   - Flooding could exacerbate electrical system issues, creating compounded risk\n   - Age of building likely affects all systems and structural integrity\n\n5. Overall Risk Classification: HIGH\n   - Multiple high-severity risks with moderate-to-high likelihood\n   - Compounding factors increase overall risk profile\n\n6. Recommended Conditions/Exclusions:\n   - Require professional electrical system inspection and upgrades\n   - Mandate flood mitigation measures (barriers, pumps, elevated equipment)\n   - Implement quarterly inspection schedule\n   - Apply higher deductible for flood and electrical damage\n   - Consider exclusion for certain flood scenarios if mitigation not implemented"
                elif "claim" in template.get("tags", []) or "claim" in template.get("name", "").lower():
                    full_text = "Dear Policyholder,\n\nThank you for submitting your claim regarding the hail damage to your vehicle. I'm pleased to inform you that after reviewing your policy, this damage is covered under your comprehensive coverage, which includes protection from weather-related incidents such as hailstorms.\n\nBased on the information provided, we will proceed with processing your claim. You'll only be responsible for your $500 deductible, and we'll cover the remaining repair costs. You may take your vehicle to any licensed auto body shop of your choice for repairs.\n\nTo move forward, please provide the following:\n1. Photos of the damage (if not already submitted)\n2. An estimate from your chosen repair facility\n3. A copy of the police report, if one was filed\n\nOnce we receive these items, we can authorize repairs and issue payment directly to the repair facility, minus your deductible.\n\nIf you have any questions about this process or need recommendations for repair facilities in your area, please don't hesitate to contact me at (555) 123-4567 or claims@insurancecompany.com.\n\nSincerely,\nClaims Representative"
                else:
                    full_text = "Based on your policy details, your auto insurance does include rental car coverage while your vehicle is being repaired after a covered accident. Your policy (#123456) provides rental reimbursement coverage of up to $30 per day for a maximum of 30 days or until your vehicle repairs are completed, whichever comes first.\n\nTo use this benefit, you'll need to:\n\n1. Make sure your vehicle repairs are due to a covered claim\n2. Notify us before renting a vehicle\n3. Keep all rental receipts for reimbursement\n\nYour comprehensive coverage will also extend to the rental car, but keep in mind that your same deductible ($1,000) would apply if there were any damages to the rental vehicle.\n\nIs there anything else you'd like to know about your rental coverage or claim process?"

                words = full_text.split()
                for i, word in enumerate(words):

                    if i > 0:
                        generated_text += " "
                    generated_text += word

                    output_container.markdown(
                        f"**Generated Output:**\n\n{generated_text}")

                    time.sleep(0.05)

                generated_text = full_text

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                generation_data = {
                    "model": model_name,
                    "prompt": st.session_state.current_prompt,
                    "generated_text": generated_text,
                    "parameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    "template_id": template.get("id", "unknown"),
                    "template_name": template.get("name", "Unnamed Template"),
                    "timestamp": current_time,
                    "task_type": template.get("task_type", "unknown"),
                }

                if "generation_history" not in st.session_state:
                    st.session_state.generation_history = []

                st.session_state.generation_history.append(generation_data)
                st.rerun()

        if "current_generation" in st.session_state:
            display_generated_text(
                st.session_state.current_generation["prompt"],
                st.session_state.current_generation["generated_text"],
                st.session_state.current_generation["model"],
                st.session_state.current_generation["parameters"]
            )

        if "generation_history" in st.session_state and st.session_state.generation_history:
            with st.expander("Generation History", expanded=False):

                history_data = [
                    {
                        "Timestamp": g["timestamp"],
                        "Model": g["model"],
                        "Template": g["template_name"],
                        "Task": g["task_type"].capitalize(),
                        "Temperature": g["parameters"]["temperature"],
                    }
                    for g in st.session_state.generation_history
                ]

                for item in history_data:
                    item["Temperature"] = str(item["Temperature"])

                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True,
                             hide_index=True)

                timestamps = [g["timestamp"]
                              for g in st.session_state.generation_history]
                selected_timestamp = st.selectbox(
                    "Select Generation", options=timestamps)

                selected_generation = next(
                    (g for g in st.session_state.generation_history if g["timestamp"] == selected_timestamp), None)

                if selected_generation and st.button("View Selected Generation"):
                    st.session_state.current_generation = selected_generation
                    st.rerun()

                if st.button("Clear History", type="secondary"):
                    st.session_state.generation_history = []
                    if "current_generation" in st.session_state:
                        del st.session_state.current_generation
                    st.success("Generation history cleared.")
                    st.rerun()

if __name__ == "__main__":

    st.set_page_config(
        page_title="Text Generation - Insurance LLM Framework",
        page_icon="🔄",
        layout="wide"
    )

    if "active_model" not in st.session_state:
        st.session_state.active_model = "llama2-7b"

    if "active_template" not in st.session_state:

        st.session_state.active_template = {
            "id": "test_template",
            "name": "Test Template",
            "description": "A test template for policy summarization",
            "format": "zero-shot",
            "task_type": "summarization",
            "insurance_domain": "general",
            "template": "Summarize the following insurance policy document:\n\n{policy_document}",
            "input_variables": ["policy_document"],
            "tags": ["policy", "summarization"]
        }

    if "model_config" not in st.session_state:
        st.session_state.model_config = {
            "temperature": 0.7,
            "max_tokens": 1000
        }

    render()
