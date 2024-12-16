import os
import sys
import pytest
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rdd"))
sys.path.append(project_path)

from RDDcounts import RDDCounts
from visualization import MatplotlibBackend, Visualizer, PlotlyBackend
from analysis import perform_pca_RDD_counts


@pytest.fixture
def mock_gnps_and_metadata(tmp_path):
    """
    Create mock GNPS and metadata files for testing.

    Parameters
    ----------
    tmp_path : pytest fixture
        Temporary path for creating test files.

    Returns
    -------
    Tuple[str, str]
        Paths to the mock GNPS and metadata files.
    """
    # Simulated GNPS data
    gnps_data = {
        "DefaultGroups": ["G1", "G4", "G1,G4", "G1,G4"],
        "UniqueFileSources": [
            "file_samp_1.mzXML|file_samp_2.mzXML",
            "file_ref1.mzXML|file_ref2.mzXML",
            "file_samp_1.mzXML|file_ref1.mzXML",
            "file_samp_2.mzXML|file_ref2.mzXML",
        ],
        "G1": [2, 0, 4, 3],
        "G2": [0, 0, 0, 0],
        "G3": [0, 0, 0, 0],
        "G4": [0, 26, 4, 3],
        "G5": [0, 0, 0, 0],
        "G6": [0, 0, 0, 0],
        "cluster index": [2, 3, 4, 5],
    }
    gnps_df = pd.DataFrame(gnps_data)
    gnps_df.columns = gnps_df.columns.str.strip()  # Clean column names
    gnps_path = tmp_path / "mock_gnps.tsv"
    gnps_df.to_csv(gnps_path, sep="\t", index=False)

    # Simulated Metadata
    metadata_data = {
        "filename": ["file_ref1.mzXML", "file_ref2.mzXML"],
        "ontology_terminal_leaf": ["complex", "plant"],
        "sample_type_group1": ["complex", "plant"],
        "sample_type_group2": ["complex", "fruit"],
        "sample_type_group3": ["complex", "fruit"],
        "sample_type_group4": ["complex", "fruit"],
        "sample_type_group5": ["complex", "fruit"],
        "sample_type_group6": ["complex", "fruit"],
        "sample_name": ["complex", "fruit"],
    }
    metadata_df = pd.DataFrame(metadata_data)
    metadata_path = tmp_path / "mock_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    return gnps_path, metadata_path

@pytest.fixture
def rdd_counts_instance(mock_gnps_and_metadata):
    """
    Create an RDDCounts instance using mock GNPS and metadata files.

    Parameters
    ----------
    mock_gnps_and_metadata : tuple
        Paths to mock GNPS and metadata files.

    Returns
    -------
    RDDCounts
        Initialized RDDCounts instance.
    """
    gnps_path, metadata_path = mock_gnps_and_metadata

    # Create the RDDCounts instance
    return RDDCounts(
        gnps_network=str(gnps_path),
        sample_types="all",
        sample_groups=["G1"],
        reference_groups=["G4"],
        levels=3,
        external_metadata=str(metadata_path),
    )

def test_plot_reference_type_distribution(rdd_counts_instance):
    """
    Test the reference type distribution visualization.
    """
    backend = MatplotlibBackend()
    visualizer = Visualizer(backend)

    # Plot reference type distribution
    fig = visualizer.plot_reference_type_distribution(
        RDD_counts_instance=rdd_counts_instance,
        level=3,
        group_by=False,
        figsize=(8, 6),
    )

    assert isinstance(fig, plt.Figure), "Output should be a Matplotlib figure"

def test_box_plot_RDD_proportions(rdd_counts_instance):
    """
    Test the box plot for RDD proportions.
    """
    backend = MatplotlibBackend()
    visualizer = Visualizer(backend)

    # Plot boxplot of RDD proportions
    fig = visualizer.box_plot_RDD_proportions(
        RDD_counts_instance=rdd_counts_instance,
        level=3,
        group_by=False,
        figsize=(8, 6),
    )

    assert isinstance(fig, plt.Figure), "Output should be a Matplotlib figure"

def test_plot_RDD_proportion_heatmap(rdd_counts_instance):
    """
    Test the heatmap visualization for RDD proportions.
    """
    backend = MatplotlibBackend()
    visualizer = Visualizer(backend)

    # Plot heatmap of proportions
    fig = visualizer.plot_RDD_proportion_heatmap(
        RDD_counts_instance=rdd_counts_instance,
        level=3,
        figsize=(10, 8),
    )

    assert isinstance(fig, plt.Figure), "Output should be a Matplotlib figure"

def test_plot_pca_results(rdd_counts_instance):
    """
    Test the PCA results scatter plot.
    """
    # Perform PCA
    pca_df, explained_variance = perform_pca_RDD_counts(
        rdd_counts_instance, level=2, apply_clr=True, n_components=2
    )

    backend = MatplotlibBackend()
    visualizer = Visualizer(backend)

    # Plot PCA results
    fig = visualizer.plot_pca_results(
        pca_df=pca_df,
        explained_variance=explained_variance,
        x_pc="PC1",
        y_pc="PC2",
        group_by=True,
        figsize=(10, 6),
    )

    assert isinstance(fig, plt.Figure), "Output should be a Matplotlib figure"

def test_plot_explained_variance():
    """
    Test the explained variance bar chart for PCA.
    """
    backend = MatplotlibBackend()
    visualizer = Visualizer(backend)

    # Mock explained variance
    explained_variance = [0.5, 0.3, 0.2]

    # Plot explained variance
    fig = visualizer.plot_explained_variance(
        explained_variance=explained_variance, figsize=(8, 6)
    )

    assert isinstance(fig, plt.Figure), "Output should be a Matplotlib figure"

def test_plot_reference_type_distribution_plotly(rdd_counts_instance):
    backend = PlotlyBackend()
    visualizer = Visualizer(backend)

    # Generate the plot
    fig = visualizer.plot_reference_type_distribution(
        rdd_counts_instance, level=3, group_by=False
    )

    # Assertions
    assert isinstance(fig, go.Figure), "Output should be a Plotly figure"
    assert len(fig.data) > 0, "Figure should contain data"

def test_box_plot_RDD_proportions_plotly(rdd_counts_instance):
    backend = PlotlyBackend()
    visualizer = Visualizer(backend)

    # Generate the plot
    fig = visualizer.box_plot_RDD_proportions(
        rdd_counts_instance, level=3, group_by=True
    )

    # Assertions
    assert isinstance(fig, go.Figure), "Output should be a Plotly figure"
    assert len(fig.data) > 0, "Figure should contain data"

def test_plot_RDD_proportion_heatmap_plotly(rdd_counts_instance):
    backend = PlotlyBackend()
    visualizer = Visualizer(backend)

    # Generate the plot
    fig = visualizer.plot_RDD_proportion_heatmap(
        rdd_counts_instance, level=3
    )

    # Assertions
    assert isinstance(fig, go.Figure), "Output should be a Plotly figure"
    assert len(fig.data) > 0, "Figure should contain data"


def test_plot_pca_results_plotly(rdd_counts_instance):
    backend = PlotlyBackend()
    visualizer = Visualizer(backend)

    # Mock PCA data
    pca_df = pd.DataFrame({
        "PC1": [1.0, 2.0, 3.0],
        "PC2": [2.0, 3.0, 4.0],
        "filename": ["file1", "file2", "file3"],
        "group": ["G1", "G2", "G1"],
    })
    explained_variance = [0.7, 0.2]

    # Generate the plot
    fig = visualizer.plot_pca_results(
        pca_df, explained_variance, x_pc="PC1", y_pc="PC2", group_by=True
    )

    # Assertions
    assert isinstance(fig, go.Figure), "Output should be a Plotly figure"
    assert len(fig.data) > 0, "Figure should contain data"


def test_plot_explained_variance_plotly():
    backend = PlotlyBackend()
    visualizer = Visualizer(backend)

    # Mock explained variance
    explained_variance = [0.7, 0.2, 0.1]

    # Generate the plot
    fig = visualizer.plot_explained_variance(explained_variance)

    # Assertions
    assert isinstance(fig, go.Figure), "Output should be a Plotly figure"
    assert len(fig.data) > 0, "Figure should contain data"

def test_plot_sankey_plotly(rdd_counts_instance, tmp_path):
    backend = PlotlyBackend()
    visualizer = Visualizer(backend)

    # Create mock color mapping file
    color_mapping_path = tmp_path / "color_mapping.csv"
    color_mapping_data = {
        "descriptor": ["type1", "type2"],
        "color_code": ["#FF5733", "#33FF57"],
    }
    pd.DataFrame(color_mapping_data).to_csv(color_mapping_path, sep=";", index=False)

    # Generate the Sankey plot
    fig = visualizer.plot_sankey(
        rdd_counts_instance, color_mapping_file=str(color_mapping_path)
    )

    # Assertions
    assert isinstance(fig, go.Figure), "Output should be a Plotly figure"
    assert len(fig.data) > 0, "Figure should contain data"