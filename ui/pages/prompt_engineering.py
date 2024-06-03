"""
Prompt Engineering Page for the Insurance LLM Framework.

This module provides the interface for creating, editing, and managing prompt templates.
"""

import streamlit as st
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import UI components
from ui.components.prompt_templates import (
    template_selector,
    display_template_info,
    template_editor,
    template_preview
)

logger = logging.getLogger(__name__)


def render():
    """Render the prompt engineering page."""
    st.title("✏️ Prompt Engineering")

    with st.expander("ℹ️ About Prompt Engineering", expanded=False):
        st.markdown("""
        This page allows you to create, edit, and manage prompt templates for insurance tasks.
        
        ### Effective Prompt Engineering
        Crafting effective prompts for insurance tasks involves:
        - **Being specific** about the task and desired output format
        - **Providing context** relevant to insurance scenarios
        - **Using examples** of good responses (few-shot learning)
        - **Specifying constraints** like regulatory requirements
        
        ### Prompt Strategies
        - **Zero-shot**: Direct instructions without examples
        - **Few-shot**: Including examples of desired inputs and outputs
        - **Chain-of-thought**: Guiding the model through reasoning steps
        - **Self-consistency**: Having the model generate multiple solutions
        """)

    # Create tabs for different prompt engineering functions
    tabs = st.tabs(
        ["Browse Templates", "Create/Edit Template", "Template Information"])

    # Model check
    if "active_model" not in st.session_state or not st.session_state.active_model:
        st.warning(
            "⚠️ No model is currently loaded. Templates will only be saved but not tested.")

    # Browse Templates Tab
    with tabs[0]:
        st.subheader("Browse Prompt Templates")

        # Template selector component
        selected_template = template_selector()

        if selected_template:
            # Display selected template preview
            template_preview(selected_template)

            # Actions for the selected template
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.button("📝 Edit Template", use_container_width=True):
                    st.session_state.editing_template = selected_template
                    st.rerun()

            with col2:
                if st.button("📋 Use Template", use_container_width=True):
                    st.session_state.active_template = selected_template
                    st.success(
                        f"Template '{selected_template['name']}' set as active")
                    logger.info(
                        f"Template '{selected_template['name']}' set as active")

            with col3:
                # Clone button to duplicate a template
                if st.button("🔄 Clone Template", use_container_width=True):
                    # Create a copy with a new name
                    new_template = selected_template.copy()
                    new_template["name"] = f"{selected_template['name']} (Copy)"
                    new_template["id"] = f"{selected_template['id']}_copy_{datetime.now().strftime('%Y%m%d%H%M%S')}"

                    # Store in session state for editing
                    st.session_state.editing_template = new_template
                    st.rerun()

    # Create/Edit Template Tab
    with tabs[1]:
        # Check if we're editing an existing template
        editing = False
        template_to_edit = None

        if "editing_template" in st.session_state and st.session_state.editing_template:
            editing = True
            template_to_edit = st.session_state.editing_template
            st.subheader(f"Edit Template: {template_to_edit['name']}")
        else:
            st.subheader("Create New Template")

        # Template editor component
        template_editor(template_to_edit)

    # Template Information Tab
    with tabs[2]:
        st.subheader("Template Information")

        selected_template_info = None

        # If we have an active template, show that
        if "active_template" in st.session_state and st.session_state.active_template:
            selected_template_info = st.session_state.active_template
        # Otherwise, if a template was selected in the browse tab
        elif selected_template:
            selected_template_info = selected_template

        if selected_template_info:
            display_template_info(selected_template_info)
        else:
            st.info(
                "Select a template to view its information or set an active template.")

    # Current active template indicator in sidebar
    with st.sidebar:
        if "active_template" in st.session_state and st.session_state.active_template:
            st.markdown("### Active Template")
            active_template = st.session_state.active_template
            st.info(f"Currently using: **{active_template['name']}**")

            # Show template format and task type
            st.caption(f"Format: {active_template['format']}")
            st.caption(f"Task: {active_template['task_type']}")

            # Quick action to clear the template
            if st.button("Clear Active Template"):
                st.session_state.active_template = None
                st.success("Active template cleared")
                st.rerun()


if __name__ == "__main__":
    # For testing the page in isolation
    st.set_page_config(
        page_title="Prompt Engineering - Insurance LLM Framework",
        page_icon="✏️",
        layout="wide"
    )

    # Initialize session state for testing
    if "active_model" not in st.session_state:
        st.session_state.active_model = "llama2-7b"

    if "active_template" not in st.session_state:
        st.session_state.active_template = None

    render()
