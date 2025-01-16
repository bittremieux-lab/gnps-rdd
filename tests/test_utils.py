import os
import sys
import pytest
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from importlib import resources

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rdd"))
sys.path.append(project_path)

from RDDcounts import RDDCounts
from utils import RDD_counts_to_wide, _load_RDD_metadata


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


def test_RDD_counts_to_wide(rdd_counts_instance):
    """
    Test the RDD_counts_to_wide function using the simulated RDDCounts instance.
    """
    # Inject mock RDD counts data
    simulated_RDD_counts = pd.DataFrame(
        {
            "filename": [
                "file_samp_1.mzXML",
                "file_samp_2.mzXML",
                "file_samp3.mzXML",
                "file_samp4.mzXML",
            ],
            "reference_type": ["type1", "type1", "type2", "type2"],
            "count": [5, 10, 20, 25],
            "level": [3, 3, 3, 3],
            "group": ["G1", "G1", "G1", "G1"],
        }
    )
    rdd_counts_instance.counts = simulated_RDD_counts

    # Expected output in wide format
    expected_wide_format = pd.DataFrame(
        {
            "type1": [5.0, 10.0, 0.0, 0.0],
            "type2": [0.0, 0.0, 20.0, 25.0],
            "group": ["G1", "G1", "G1", "G1"],
        },
        index=[
            "file_samp_1.mzXML",
            "file_samp_2.mzXML",
            "file_samp3.mzXML",
            "file_samp4.mzXML",
        ],
    )
    expected_wide_format.index.name = "filename"

    # Call the function with level 3
    result = RDD_counts_to_wide(rdd_counts_instance.counts, level=3)

    # Sort index and columns for comparison
    result = result.sort_index().sort_index(axis=1)
    expected_wide_format = expected_wide_format.sort_index().sort_index(axis=1)

    # Check if the result matches the expected output
    pd.testing.assert_frame_equal(result, expected_wide_format)

    # Edge case: Test with a non-existent level
    with pytest.raises(ValueError, match="No data available for level 5"):
        RDD_counts_to_wide(rdd_counts_instance.counts, level=5)


def test_invalid_metadata_format():
    with pytest.raises(
        ValueError, match="External metadata file must be a CSV, TSV, or TXT."
    ):
        _load_RDD_metadata("invalid_file_format.doc")


def test_missing_metadata_file():
    with pytest.raises(
        FileNotFoundError, match="External metadata file 'missing_file.csv' not found."
    ):
        _load_RDD_metadata("missing_file.csv")


@pytest.fixture
def mock_resources(monkeypatch):
    def mock_open_text(package, resource):
        from io import StringIO

        mock_data = "filename\tsample_type\nfile1\tcomplex\nfile2\tplant"
        return StringIO(mock_data)

    monkeypatch.setattr(resources, "open_text", mock_open_text)


def test_load_external_metadata(tmp_path):
    # Create a mock external metadata file
    metadata_file = tmp_path / "external_metadata.csv"
    metadata_file.write_text("filename,sample_type\nfile1,complex\nfile2,plant")

    metadata = _load_RDD_metadata(str(metadata_file))
    assert not metadata.empty
    assert "filename" in metadata.columns


def test_invalid_metadata_format(tmp_path):
    invalid_file = tmp_path / "metadata.invalid"
    invalid_file.write_text("Invalid content")

    with pytest.raises(ValueError):
        _load_RDD_metadata(str(invalid_file))


def test_missing_metadata_file(tmp_path):
    missing_file = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        _load_RDD_metadata(str(missing_file))
