"""
Street name cleaning and standardization utilities.
"""

import pandas as pd
from typing import Optional

from .config import SUFFIX_MAP


def clean_street_name(
    name_series: pd.Series,
    suffix_series: Optional[pd.Series] = None,
    prefix_series: Optional[pd.Series] = None
) -> pd.Series:
    """
    Standardize street names by combining components and applying canonical abbreviations.

    Combines prefix, name, and suffix into a single standardized string. Applies
    uppercase formatting and replaces long street name versions with short abbreviations.

    Parameters
    ----------
    name_series : pd.Series
        Series containing street names.
    suffix_series : pd.Series, optional
        Series containing street suffixes.
    prefix_series : pd.Series, optional
        Series containing directional prefixes.

    Returns
    -------
    pd.Series
        Series of standardized street names.
    """
    full_name = name_series.fillna('')
    
    if prefix_series is not None:
        full_name = prefix_series.fillna('') + ' ' + full_name
        
    if suffix_series is not None:
        full_name = full_name + ' ' + suffix_series.fillna('')

    full_name = full_name.str.upper().str.strip()
    
    for long_ver, short_ver in SUFFIX_MAP.items():
        full_name = full_name.str.replace(fr'\b{long_ver}\b', short_ver, regex=True)

    full_name = full_name.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return full_name

