# Standard library imports
import os
import pkg_resources
from importlib import resources
from typing import List, Optional

# Third-party imports
import pandas as pd


def _load_RDD_metadata(external_metadata: Optional[str] = None) -> pd.DataFrame:
    """
    Reads ontology and metadata from the default file or an external file.

    Parameters
    ----------
    external_metadata : str, optional
        Path to an external metadata file. If provided, this file will be loaded.
        If None, the default internal metadata will be used.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing ontology and metadata.

    Raises
    ------
    FileNotFoundError
        If the external file path does not exist.
    ValueError
        If the external file is not in a valid format (must be a CSV/TSV).
    """
    if external_metadata:
        # Load user-provided metadata
        if not external_metadata.lower().endswith((".csv", ".tsv", ".txt")):
            raise ValueError("External metadata file must be a CSV, TSV, or TXT.")

        # Detect separator based on file extension
        sep = "\t" if external_metadata.lower().endswith((".tsv", ".txt")) else ","

        try:
            return pd.read_csv(external_metadata, sep=sep)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"External metadata file '{external_metadata}' not found."
            )
    else:
        # Default behavior: load internal metadata
        try:
            with resources.open_text(
                "data", "foodomics_multiproject_metadata.txt"
            ) as stream:
                reference_metadata = pd.read_csv(stream, sep="\t")
        except (ModuleNotFoundError, ImportError):
            stream = pkg_resources.resource_stream(
                __name__, "data/foodomics_multiproject_metadata.txt"
            )
            reference_metadata = pd.read_csv(stream, sep="\t")
        return reference_metadata


def _load_sample_types(
    reference_metadata: pd.DataFrame, simple_complex: str = "all"
) -> pd.DataFrame:
    """
    Filters metadata by simple, complex, or all types of references.

    Parameters
    ----------
    reference_metadata : pd.DataFrame
        The metadata DataFrame to filter.
    simple_complex : str, optional
        One of 'simple', 'complex', or 'all'.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame of ontology.
    """
    if simple_complex != "all":
        reference_metadata = reference_metadata[
            reference_metadata["simple_complex"] == simple_complex
        ]

    col_sample_types = ["sample_name"] + [f"sample_type_group{i}" for i in range(1, 7)]
    return reference_metadata[["filename", *col_sample_types]].set_index("filename")


def _validate_groups(gnps_network: pd.DataFrame, groups_included: List[str]) -> None:
    """
    Validates that the provided group names exist in the GNPS network data.

    Parameters
    ----------
    gnps_network : pd.DataFrame
        The GNPS network DataFrame to validate against.
    groups_included : List[str]
        The groups to validate.

    Raises
    ------
    ValueError
        If any of the group names in `groups_included` are invalid.
    """
    valid_groups = set(gnps_network["DefaultGroups"].unique())

    # Check included groups
    invalid_included_groups = set(groups_included) - valid_groups
    if len(invalid_included_groups) > 0:
        raise ValueError(
            f"The following groups in groups_included are invalid: {invalid_included_groups}"
        )


def RDD_counts_to_wide(RDD_counts: pd.DataFrame, level: int = None) -> pd.DataFrame:
    """
    Convert the RDD counts dataframe from long to wide format for a specific
    ontology level, with 'group' as part of the columns. If the data is already
    filtered by level, no level needs to be passed.

    Parameters
    ----------
    RDD_counts : pd.DataFrame
        A long-format DataFrame with columns ['filename', 'reference_type', 'count',
        'level', 'group'] representing RDD counts across different ontology
        levels and groups.
    level : int, optional
        The ontology level to filter by before converting to wide format. If
        None, the level is inferred from the data. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A wide-format DataFrame where each combination of 'reference_type' and
        'group' becomes a column, and rows are indexed by 'filename'.

    Raises
    ------
    ValueError
        If multiple levels are found in the data and `level` is not specified,
        or if no data is available for the specified level.
    """
    # If level is not specified, infer the level from the data
    if level is None:
        levels_in_data = RDD_counts["level"].unique()
        if len(levels_in_data) > 1:
            raise ValueError(
                "Multiple levels found in the data. Please specify a level to convert to wide format."
            )
        level = levels_in_data[0]  # If the data is already filtered, use that level

    # Filter the RDD counts dataframe by the specified level, if not already filtered
    filtered_RDD_counts = RDD_counts[RDD_counts["level"] == level]

    if filtered_RDD_counts.empty:
        raise ValueError(f"No data available for level {level}")

    # Pivot the filtered dataframe to wide format with 'reference_type' and 'group' as columns
    RDD_counts_wide = filtered_RDD_counts.pivot_table(
        index="filename", columns="reference_type", values="count", fill_value=0
    )
    group_df = (
        filtered_RDD_counts[["filename", "group"]]
        .drop_duplicates()
        .set_index("filename")
    )
    wide_format_RDD_counts = RDD_counts_wide.join(group_df)

    return wide_format_RDD_counts


def calculate_proportions(RDD_counts: pd.DataFrame, level: int = None) -> pd.DataFrame:
    """
    Calculate the proportion of each reference type within each sample for a given
    level.

    Parameters
    ----------
    RDD_counts : pd.DataFrame
        A long-format DataFrame with columns ['filename', 'reference_type', 'count',
        'level', 'group'] representing RDD counts across different ontology
        levels and groups.
    level : int, optional
        The ontology level to filter by before calculating proportions. If None,
        the level is inferred from the data. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A wide-format DataFrame where each reference type column contains the
        proportion of that reference type within each sample (row). Rows are indexed
        by 'filename', and proportions sum to 1 for each sample.

    Raises
    ------
    ValueError
        If multiple levels are found in the data and `level` is not specified.
    """
    # If level is not specified, infer the level from the data
    if level is None:
        levels_in_data = RDD_counts["level"].unique()
        if len(levels_in_data) > 1:
            raise ValueError(
                "Multiple levels found in the data. Please specify a level to calculate proportions."
            )
        level = levels_in_data[0]  # If the data is already filtered, use that level

    # Use the existing function to convert to wide format
    df_wide = RDD_counts_to_wide(RDD_counts, level)

    # Identify numeric columns (reference type counts)
    numeric_cols = df_wide.select_dtypes(include=[float, int]).columns

    # Calculate proportions across reference types (columns) for each sample (row)
    df_proportions = df_wide.copy()
    df_proportions[numeric_cols] = (
        df_proportions[numeric_cols]
        .div(df_proportions[numeric_cols].sum(axis=1), axis=0)
        .fillna(0)
    )

    return df_proportions
