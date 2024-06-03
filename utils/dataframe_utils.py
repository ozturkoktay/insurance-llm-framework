def prepare_dataframe_for_display(df):
    """
    Convert DataFrame columns to appropriate types for Arrow serialization.
    
    Args:
        df: pandas DataFrame to prepare
    
    Returns:
        DataFrame with compatible types for Arrow serialization
    """
    # Create a copy to avoid modifying the original
    df_display = df.copy()
    
    # Convert object columns that might contain mixed types to strings
    for col in df_display.select_dtypes(include=['object']).columns:
        df_display[col] = df_display[col].astype(str)
    
    # Convert any other problematic types
    for col in df_display.columns:
        if df_display[col].dtype.name not in ['int64', 'float64', 'bool', 'datetime64[ns]', 'string', 'str']:
            df_display[col] = df_display[col].astype(str)
    
    return df_display 