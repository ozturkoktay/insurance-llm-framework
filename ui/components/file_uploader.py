"""
File upload components for the Insurance LLM Framework.

This module provides UI components for uploading and managing insurance documents.
"""

import streamlit as st
import pandas as pd
import os
from typing import Dict, Any, Optional, List, Tuple, BinaryIO
import logging
from pathlib import Path
import base64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def file_uploader(
    file_type: str = "text",
    allowed_types: Optional[List[str]] = None,
    default_dir: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Display a file uploader with options to upload a file or select a sample.

    Args:
        file_type: Type of file to upload (e.g., "text", "policy", "claim")
        allowed_types: List of allowed file extensions
        default_dir: Default directory to look for sample files

    Returns:
        Tuple containing the text content and the filename
    """

    if allowed_types is None:
        allowed_types = ["txt", "pdf", "docx", "json"]

    sample_dir = None
    if default_dir:
        sample_dir = default_dir
    elif file_type == "policy":
        sample_dir = "data/policies"
    elif file_type == "claim":
        sample_dir = "data/claims"
    elif file_type == "communication":
        sample_dir = "data/communications"

    sample_files = []
    if sample_dir and os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir)
                        if any(f.endswith(ext) for ext in allowed_types)]

    tab1, tab2, tab3 = st.tabs(["Upload File", "Sample Files", "Enter Text"])

    file_content = None
    file_name = None

    with tab1:
        uploaded_file = st.file_uploader(
            f"Upload {file_type.capitalize()} File",
            type=allowed_types,
            help=f"Upload a {file_type} file to process"
        )

        if uploaded_file is not None:
            try:

                file_content = uploaded_file.read().decode("utf-8")
                file_name = uploaded_file.name
                st.success(f"Uploaded: {file_name}")

                with st.expander("Preview Content", expanded=True):
                    st.text_area("File Content", value=file_content[:1000] + (
                        "..." if len(file_content) > 1000 else ""), height=200, disabled=True)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                logger.error(f"Error reading uploaded file: {str(e)}")

    with tab2:
        if sample_files:
            selected_sample = st.selectbox(
                f"Select Sample {file_type.capitalize()} File",
                options=sample_files,
                help=f"Choose a sample {file_type} file from the available options"
            )

            if selected_sample:
                try:
                    sample_path = os.path.join(sample_dir, selected_sample)
                    with open(sample_path, "r") as f:
                        file_content = f.read()
                    file_name = selected_sample
                    st.success(f"Loaded sample: {file_name}")

                    with st.expander("Preview Content", expanded=True):
                        st.text_area("File Content", value=file_content[:1000] + (
                            "..." if len(file_content) > 1000 else ""), height=200, disabled=True)
                except Exception as e:
                    st.error(f"Error reading sample file: {str(e)}")
                    logger.error(
                        f"Error reading sample file {sample_path}: {str(e)}")
        else:
            st.info(f"No sample {file_type} files available")

    with tab3:
        manual_text = st.text_area(
            f"Enter {file_type.capitalize()} Text",
            height=300,
            help=f"Paste or type {file_type} text directly"
        )

        if manual_text:
            file_content = manual_text
            file_name = f"manual_{file_type}.txt"
            st.success("Text entered successfully")

    return file_content, file_name

def save_uploaded_file(
    uploaded_file: BinaryIO,
    save_dir: str,
    file_name: Optional[str] = None
) -> Optional[str]:
    """
    Save an uploaded file to the specified directory.

    Args:
        uploaded_file: The uploaded file
        save_dir: Directory to save the file
        file_name: Optional name to use when saving the file

    Returns:
        The path to the saved file, or None if saving failed
    """
    try:

        os.makedirs(save_dir, exist_ok=True)

        if file_name is None:
            file_name = uploaded_file.name

        file_path = os.path.join(save_dir, file_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Saved uploaded file to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return None

def display_document_library(doc_dir: str,
                             filter_extension: Optional[List[str]] = None,
                             on_select: Optional[callable] = None):
    """
    Display a document library with available documents.

    Args:
        doc_dir: Directory containing the documents
        filter_extension: Optional list of file extensions to filter by
        on_select: Optional callback function when a document is selected
    """

    if not os.path.exists(doc_dir):
        st.warning(f"Document directory not found: {doc_dir}")
        return None

    files = os.listdir(doc_dir)

    if filter_extension:
        files = [f for f in files if any(f.lower().endswith(
            ext.lower()) for ext in filter_extension)]

    files.sort()

    if not files:
        st.info("No documents available")
        return None

    file_data = []
    for file in files:
        file_path = os.path.join(doc_dir, file)
        file_size = os.path.getsize(file_path)
        file_modified = os.path.getmtime(file_path)

        file_data.append({
            "File Name": file,
            "Size (KB)": f"{file_size / 1024:.1f}",
            "Last Modified": pd.to_datetime(file_modified, unit='s').strftime("%Y-%m-%d %H:%M"),
            "Type": file.split(".")[-1].upper()
        })

    docs_df = pd.DataFrame(file_data)

    st.dataframe(docs_df, hide_index=True, use_container_width=True)

    selected_doc = st.selectbox("Select Document", options=files)

    if selected_doc and on_select:
        on_select(os.path.join(doc_dir, selected_doc))

    return selected_doc if selected_doc else None

def document_viewer(file_path: str, max_height: int = 400):
    """
    Display a document viewer for the specified file.

    Args:
        file_path: Path to the document file
        max_height: Maximum height for the viewer in pixels
    """
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    file_extension = file_path.lower().split(".")[-1]

    if file_extension in ["txt", "md", "json"]:
        try:
            with open(file_path, "r") as f:
                content = f.read()

            st.text_area("Document Content", value=content,
                         height=max_height, disabled=True)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    elif file_extension == "pdf":

        try:
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            pdf_display = f"""
                <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="{max_height}" type="application/pdf"></iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying PDF: {str(e)}")

    elif file_extension in ["jpg", "jpeg", "png", "gif"]:

        try:
            st.image(file_path, caption=os.path.basename(file_path))
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")

    else:

        st.info(f"Preview not available for {file_extension.upper()} files")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        st.download_button(
            label="Download File",
            data=file_bytes,
            file_name=os.path.basename(file_path),
            mime=f"application/{file_extension}"
        )
