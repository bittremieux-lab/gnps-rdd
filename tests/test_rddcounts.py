import os
import sys
import pytest
import pandas as pd

# Add the path to the 'rdd' directory to the system path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rdd"))
sys.path.append(project_path)

from RDDcounts import RDDCounts


@pytest.fixture
def mock_gnps_and_metadata(tmp_path):
    # Create mock GNPS file
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

    # Create mock metadata file
    metadata = pd.DataFrame(
        {
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
    )
    metadata_path = tmp_path / "mock_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    return gnps_path, metadata_path


def test_initialization(mock_gnps_and_metadata):
    gnps_path, metadata_path = mock_gnps_and_metadata

    # Initialize RDDCounts
    rdd_counts = RDDCounts(
        gnps_network=str(gnps_path),
        sample_types="all",
        sample_groups=["G1"],
        reference_groups=["G4"],
        levels=2,
        external_metadata=str(metadata_path),
    )

    # Check the attributes
    assert isinstance(rdd_counts.gnps_network, pd.DataFrame)
    assert isinstance(rdd_counts.RDD_metadata, pd.DataFrame)
    assert rdd_counts.levels == 2


def test_filename_level_counts(mock_gnps_and_metadata):
    gnps_path, metadata_path = mock_gnps_and_metadata

    # Initialize RDDCounts
    rdd_counts = RDDCounts(
        gnps_network=str(gnps_path),
        sample_types="all",
        sample_groups=["G1"],
        reference_groups=["G4"],
        levels=6,
        external_metadata=str(metadata_path),
    )

    # Get file-level counts
    file_counts = rdd_counts._get_filename_level_RDD_counts()

    # Verify the output
    assert not file_counts.empty, "File-level counts should not be empty"
    assert "filename" in file_counts.columns
    assert "reference_type" in file_counts.columns
    assert "count" in file_counts.columns


@pytest.mark.parametrize(
    "reference_types, level, expected_count",
    [
        (["complex"], 1, 1),
        (["fruit"], 2, 1),
        (None, 1, 2),
    ],
)
def test_filter_counts(mock_gnps_and_metadata, reference_types, level, expected_count):
    gnps_path, metadata_path = mock_gnps_and_metadata

    rdd_counts = RDDCounts(
        gnps_network=str(gnps_path),
        sample_types="all",
        sample_groups=["G1"],
        reference_groups=["G4"],
        levels=6,
        external_metadata=str(metadata_path),
    )

    # Filter counts
    filtered_counts = rdd_counts.filter_counts(
        reference_types=reference_types, level=level
    )

    # Validate filtering logic
    assert not filtered_counts.empty, "Filtered counts should not be empty"
    assert sum(filtered_counts["count"]) == expected_count
    if reference_types:
        assert all(
            filtered_counts["reference_type"].isin(reference_types)
        ), "Filtered types mismatch"


def test_update_groups(mock_gnps_and_metadata, tmp_path):
    gnps_path, metadata_path = mock_gnps_and_metadata

    # Prepare new metadata with updated groups
    updated_metadata = pd.DataFrame(
        {
            "filename": ["file_samp_1.mzXML", "file_samp_2.mzXML"],
            "new_group": ["group1", "group2"],
        }
    )
    updated_metadata_path = tmp_path / "updated_metadata.csv"
    updated_metadata.to_csv(updated_metadata_path, index=False)

    rdd_counts = RDDCounts(
        gnps_network=str(gnps_path),
        sample_types="all",
        sample_groups=["G1"],
        reference_groups=["G4"],
        levels=2,
        external_metadata=str(metadata_path),
    )

    # Update groups
    rdd_counts.update_groups(
        metadata_file=str(updated_metadata_path), merge_column="new_group"
    )
    print(rdd_counts.sample_metadata)
    # Validate group updates
    assert all(
        rdd_counts.counts["group"].isin(["group1", "group2"])
    ), "Group update failed"


def test_generate_RDDflows(mock_gnps_and_metadata):
    gnps_path, metadata_path = mock_gnps_and_metadata

    rdd_counts = RDDCounts(
        gnps_network=str(gnps_path),
        sample_types="all",
        sample_groups=["G1"],
        reference_groups=["G4"],
        levels=2,
        external_metadata=str(metadata_path),
    )

    # Generate flows and processes
    flows, processes = rdd_counts.generate_RDDflows()
    # Validate structure
    assert not flows.empty, "Flows should not be empty"
    assert not processes.empty, "Processes should not be empty"
    assert (
        "source" in flows.columns
        and "target" in flows.columns
        and "value" in flows.columns
    )
    assert processes.index.name == "id" and "level" in processes.columns
