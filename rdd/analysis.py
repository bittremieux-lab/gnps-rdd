# analysis.py

# Standard library imports
from typing import List, Tuple, Optional

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skbio.stats.composition import clr

# Internal imports
from RDDcounts import RDDCounts
from utils import RDD_counts_to_wide


def perform_pca_RDD_counts(
    RDD_counts_instance: RDDCounts,
    level: int = 3,
    n_components: int = 3,
    apply_clr: bool = True,
    reference_types: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Perform PCA on RDD counts data using the RDDCounts instance, with an
    option for CLR transformation.

    Parameters
    ----------
    RDD_counts_instance : RDDCounts
        The instance of the RDDCounts class containing RDD counts data.
    level : int, optional
        Ontology level to filter reference types, by default 3.
    n_components : int, optional
        Number of principal components to calculate, by default 3.
    apply_clr : bool, optional
        Whether to apply the CLR (Centered Log-Ratio) transformation before
        PCA, by default True.
    reference_types : list of str, optional
        List of specific reference types to include in the analysis. If None, all
        reference types are included.

    Returns
    -------
    Tuple[pd.DataFrame, List[float]]
        A tuple containing:
        - pd.DataFrame: DataFrame with PCA scores, filenames, and merged
        sample metadata.
        - List[float]: Explained variance ratios of the principal components.

    Notes
    -----
    - The CLR transformation is applied to the numeric data after adding 1 to
    avoid issues with zeros.
    """
    # Step 1: Filter counts by level
    RDD_counts_filtered = RDD_counts_instance.filter_counts(level=level)

    # Step 2: Filter reference types if specified by the user
    if reference_types is not None:
        RDD_counts_filtered = RDD_counts_filtered[
            RDD_counts_filtered["reference_type"].isin(reference_types)
        ]

    # Step 3: Convert to wide format for PCA
    RDD_counts_wide = RDD_counts_to_wide(RDD_counts_filtered, level=level)

    # Step 4: Extract numeric columns (reference types) for PCA
    reference_type_columns = RDD_counts_wide.select_dtypes(include=[np.number]).columns
    features = RDD_counts_wide[reference_type_columns].values

    # Step 5: Apply CLR transformation if selected
    if apply_clr:
        features = clr(features + 1)  # Add 1 to avoid issues with zeros, then apply CLR

    # Step 6: Standardize the data
    features_scaled = StandardScaler().fit_transform(features)

    # Step 7: Perform PCA
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(features_scaled)
    explained_variance = pca.explained_variance_ratio_

    # Step 8: Create a dataframe with PCA results
    pca_df = pd.DataFrame(pca_scores, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df["filename"] = RDD_counts_wide.index

    # Merge with sample metadata
    sample_metadata = RDD_counts_instance.sample_metadata
    pca_df = pd.merge(pca_df, sample_metadata, on="filename", how="left")

    return pca_df, explained_variance
