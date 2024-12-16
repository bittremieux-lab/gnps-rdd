# RDDcounts.py

# Standard library imports
import os
from typing import List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Internal imports
from utils import _load_RDD_metadata, _load_sample_types, _validate_groups


class RDDCounts:
    def __init__(
        self,
        gnps_network: str,
        sample_types: str,
        sample_groups: List[str],
        reference_groups: List[str],
        levels: int = 6,
        external_metadata: Optional[str] = None,
    ) -> None:
        """
        Initializes the RDDCounts object and automatically creates the RDD counts
        for all levels and all reference types.

        Parameters
        ----------
        gnps_network : str
            Path to the TSV file generated from classical molecular networking.
        sample_types : str
            One of 'simple', 'complex', or 'all', indicating which sample types to
            include in the RDD counts.
        sample_groups : list of str
            List of groups representing study spectrum files to include in the
            analysis.
        reference_groups : list of str
            List of groups representing reference spectrum files to include in the
            analysis.
        levels : int, optional
            Number of ontology levels to calculate RDD counts for, by default 6.

        external_metadata : str, optional
            Path to an external metadata file. If None, the default internal metadata is used.

        Attributes
        ----------
        gnps_network : pd.DataFrame
            Dataframe containing the GNPS network data from the specified TSV file.
        sample_types : pd.DataFrame
            Dataframe containing the sample type information, filtered by the
            specified `sample_types` parameter.
        sample_groups : list of str
            List of study group names.
        reference_groups : list of str
            List of reference group names.
        levels : int
            The number of ontology levels.
        RDD_metadata : pd.DataFrame
            Metadata with ontology, including sample names,
            descriptions, and other attributes.
        sample_metadata : pd.DataFrame
            Metadata for the samples, including filenames and group information.
        counts : pd.DataFrame
            A DataFrame of the RDD counts across different levels, grouped by
            filename and sample type.
        """
        # Load GNPS network data
        self.gnps_network = pd.read_csv(gnps_network, sep="\t")
        self.RDD_metadata = _load_RDD_metadata(external_metadata=external_metadata)
        self.sample_types = _load_sample_types(self.RDD_metadata, sample_types)
        self.sample_groups = sample_groups
        self.reference_groups = reference_groups
        self.levels = levels

        # Validate group names
        _validate_groups(self.gnps_network, self.sample_groups)
        _validate_groups(self.gnps_network, self.reference_groups)
        # Generate sample metadata and counts
        self.sample_metadata = self._get_sample_metadata()
        self.file_level_counts = self._get_filename_level_RDD_counts()
        self.counts = self.create_RDD_counts_all_levels()

    def _get_sample_metadata(self) -> pd.DataFrame:
        """
        Extracts filenames and groups from the study groups (sample_groups) in the
        GNPS network dataframe.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing filenames and their corresponding groups.
        """
        df_filtered = self.gnps_network[
            ~self.gnps_network["DefaultGroups"].str.contains(",")
        ]
        df_selected = df_filtered[df_filtered["DefaultGroups"].isin(self.sample_groups)]
        df_exploded_files = df_selected.assign(
            UniqueFileSources=df_selected["UniqueFileSources"].str.split("|")
        ).explode("UniqueFileSources")
        filenames_df = df_exploded_files[["DefaultGroups", "UniqueFileSources"]].rename(
            columns={"DefaultGroups": "group", "UniqueFileSources": "filename"}
        )
        return filenames_df.drop_duplicates().reset_index(drop=True)

    def _get_filename_level_RDD_counts(self) -> pd.DataFrame:
        """
        Generates a table of RDD counts at the filename level. It filters the GNPS network
         based on the provided sample and
        reference groups.
        Returns
        -------
        pd.DataFrame
            A DataFrame containing RDD counts at the filename level, structured with
            columns for 'filename', 'reference_type', 'count', and 'level'.
        """
        groups = {f"G{i}" for i in range(1, 7)}
        groups_excluded = list(
            groups - set([*self.sample_groups, *self.reference_groups])
        )

        # Filter GNPS network based on group criteria
        df_selected = self.gnps_network[
            (self.gnps_network[self.sample_groups] > 0).all(axis=1)
            & (self.gnps_network[self.reference_groups] > 0).any(axis=1)
            & (self.gnps_network[groups_excluded] == 0).all(axis=1)
        ].copy()

        # Explode the 'UniqueFileSources' to create individual rows for each filename
        df_exploded = df_selected.assign(
            filename=df_selected["UniqueFileSources"].str.split("|")
        ).explode("filename")

        # Create a new dataframe with the necessary columns
        df_new = df_exploded[["filename", "cluster index"]].copy()

        # Filter for samples and reference using metadata and create separate dataframes
        sample_filenames = set(self.sample_metadata["filename"])
        df_new["sample"] = df_new["filename"].isin(sample_filenames)

        # Separate samples and reference based on the sample flag
        samples_df = df_new[df_new["sample"] == True][["filename", "cluster index"]]
        RDD_df = df_new[df_new["sample"] == False][
            ["filename", "cluster index"]
        ].rename(columns={"filename": "RDD_filename"})

        # Reindex RDD dataframe to map refernece types with their corresponding sample names
        RDD_df["sample_name"] = RDD_df["RDD_filename"].map(
            self.sample_types["sample_name"]
        )

        # Filter out rows where 'sample_name' is NaN (in case some reference don't have corresponding sample names)
        RDD_df = RDD_df.dropna(subset=["sample_name"])

        # Merge samples_df and the updated RDD_df on 'cluster index'
        merged_df = pd.merge(
            samples_df,
            RDD_df[["sample_name", "cluster index"]],
            on="cluster index",
            how="inner",
        )

        RDD_counts_file_level = (
            merged_df.groupby(["filename", "sample_name"]).size().unstack(fill_value=0)
        )

        # Return the counts
        RDD_counts_file_level_long = RDD_counts_file_level.reset_index().melt(
            id_vars="filename", var_name="reference_type", value_name="count"
        )
        RDD_counts_file_level_long["level"] = 0

        return RDD_counts_file_level_long

    def create_RDD_counts_all_levels(self) -> pd.DataFrame:
        """
        Generates RDD counts across all ontology levels and compiles them into a
        single DataFrame. This function creates level-specific counts by grouping the
        file-level counts and applies a filter for samples appearing less frequent than
        water.Returns the data in long format.

        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame containing RDD counts across all levels,
            including 'filename', 'reference_type', 'count', 'level', and 'group' columns.
        """
        RDD_counts_file_level = self.file_level_counts
        RDD_counts_all_levels = [
            RDD_counts_file_level
        ]  # Initialize a list for storing data at all levels
        RDD_counts_file_level_sample_types = RDD_counts_file_level.merge(
            self.sample_types, left_on="reference_type", right_on="sample_name"
        ).drop_duplicates()

        sample_metadata_map = self.sample_metadata.set_index("filename")[
            "group"
        ].to_dict()  # Create the mapping once

        for level in range(1, self.levels + 1):
            # Group and pivot to wide format
            RDD_counts_level = (
                RDD_counts_file_level_sample_types.groupby(
                    ["filename", f"sample_type_group{level}"]
                )["count"]
                .sum()
                .reset_index()
            )
            wide_format_counts = RDD_counts_level.pivot_table(
                index="filename",
                columns=f"sample_type_group{level}",
                values="count",
                fill_value=0,
            ).reset_index()

            # Compare with water and filter
            if "water" in wide_format_counts.columns:
                water_counts = wide_format_counts["water"]
                # Apply the condition across all columns except 'filename' and 'water'
                columns_to_modify = wide_format_counts.columns.difference(
                    ["filename", "water"]
                )
                wide_format_counts.loc[:, columns_to_modify] = wide_format_counts.loc[
                    :, columns_to_modify
                ].where(
                    wide_format_counts.loc[:, columns_to_modify].gt(
                        water_counts, axis=0
                    ),
                    0,
                )

                wide_format_counts = wide_format_counts.drop(columns=["water"])

            # Drop columns where all values are 0
            wide_format_counts = wide_format_counts.loc[
                :, (wide_format_counts != 0).any(axis=0)
            ]

            # Melt back to long format
            RDD_counts_level = wide_format_counts.melt(
                id_vars="filename", var_name="reference_type", value_name="count"
            )
            RDD_counts_level["level"] = level

            RDD_counts_all_levels.append(
                RDD_counts_level
            )  # Append to the list instead of concatenating each time

        RDD_counts_all_levels = pd.concat(RDD_counts_all_levels, ignore_index=True)

        # Map group information from the sample_metadata to the final DataFrame
        RDD_counts_all_levels["group"] = RDD_counts_all_levels["filename"].map(
            sample_metadata_map
        )

        # Cast 'count' as an integer
        RDD_counts_all_levels["count"] = RDD_counts_all_levels["count"].astype(int)

        return RDD_counts_all_levels

    def filter_counts(
        self, reference_types: Optional[List[str]] = None, level: int = 3
    ) -> pd.DataFrame:
        """
        Filters the RDD counts based on reference types and ontology level.

        Parameters
        ----------
        reference_types : list of str, optional
            List of reference types to filter by. If None, all reference types at the specified
            level are included. Defaults to None.
        level : int, optional
            The ontology level to filter the RDD counts by. Defaults to 3.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered RDD counts with columns: filename,
            reference type, level, count, and group.
        """
        if self.counts is None:
            raise ValueError(
                "RDD counts have not been created yet. Call create() first."
            )
        if reference_types is None:
            filtered_df = self.counts[self.counts["level"] == level]
            return filtered_df
        filtered_df = self.counts[
            (self.counts["reference_type"].isin(reference_types))
            & (self.counts["level"] == level)
        ]
        return filtered_df

    def update_groups(self, metadata_file: str, merge_column: str) -> None:
        """
        Updates the 'group' column in the RDD counts and sample_metadata
        DataFrames based on user-provided metadata.

        Parameters
        ----------
        metadata_file : str
            Path to the metadata file (CSV or TSV) containing updated group
            information.
        merge_column : str
            The column in the metadata file to use for updating the group information.

        Raises
        ------
        ValueError
            If the metadata file is not a valid CSV or TSV file or if the necessary
            columns are missing.
        """
        # Detect file extension to determine the separator
        file_extension = os.path.splitext(metadata_file)[1].lower()
        if file_extension == ".csv":
            metadata = pd.read_csv(metadata_file)
        elif file_extension in [".tsv", ".txt"]:
            metadata = pd.read_csv(metadata_file, sep="\t")
        else:
            raise ValueError("Metadata file must be either a CSV or TSV.")

        # Ensure that the metadata contains the necessary columns
        if not {"filename", merge_column}.issubset(metadata.columns):
            raise ValueError(
                f"Metadata file must contain 'filename' and '{merge_column}' columns."
            )

        # Create a mapping from filename to new group
        filename_to_group = metadata.set_index("filename")[merge_column].to_dict()

        # Update the 'group' column in counts
        self.counts["group"] = (
            self.counts["filename"].map(filename_to_group).fillna(self.counts["group"])
        )

        # Update the sample_metadata
        self.sample_metadata["group"] = (
            self.sample_metadata["filename"]
            .map(filename_to_group)
            .fillna(self.sample_metadata["group"])
        )
    def generate_RDDflows(
    self, 
    max_hierarchy_level: Optional[int] = None, 
    filename_filter: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates RDD flows and processes across ontology levels, with options for 
        filtering by a specific sample and specifying a maximum hierarchy level.

        Parameters
        ----------
        max_hierarchy_level : int, optional
            The maximum level to calculate flows up to. Defaults to the instance's levels.
        filename_filter : str, optional
            A specific sample filename to filter by. If None, the full counts dataframe is used.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing:
            - flows: DataFrame with source, target, and value columns.
            - processes: DataFrame with unique nodes across the levels.
        """
        # Use provided max_hierarchy_level or default to the instance's levels
        max_level = max_hierarchy_level if max_hierarchy_level is not None else self.levels

        # Filter counts by filename if a filter is specified
        if filename_filter:
            counts = self.counts[self.counts["filename"] == filename_filter]
        else:
            counts = self.counts

        flows = []
        
        for i in range(1, max_level):
            # Set source and target level for the current iteration
            source_level = i
            target_level = i + 1

            # Group RDD counts at target level
            target_counts = (
                counts[counts['level'] == target_level]
                .groupby('reference_type')['count']
                .sum()
                .reset_index()
            )

            # Merge with sample_types to find corresponding source level
            merged_df = pd.merge(
                target_counts,
                self.sample_types,
                left_on='reference_type',
                right_on=f"sample_type_group{target_level}"
            )[[f"sample_type_group{source_level}", 'reference_type', 'count']].drop_duplicates()

            # Rename columns for source-target relationship and add levels for uniqueness
            flow = merged_df.rename(columns={f"sample_type_group{source_level}": "source", "reference_type": "target"})
            flow["source"] = flow["source"] + f"_{source_level}"
            flow["target"] = flow["target"] + f"_{target_level}"
            flow.rename(columns={"count": "value"}, inplace=True)

            flows.append(flow)

        # Concatenate flows into a single DataFrame
        flows_df = pd.concat(flows, ignore_index=True)

        # Build processes from unique nodes in flows
        all_nodes = pd.concat([flows_df["source"], flows_df["target"]]).unique()
        processes_df = pd.DataFrame({
            "id": all_nodes,
            "level": [int(node.split('_')[-1]) for node in all_nodes]
        }).set_index("id")

        return flows_df, processes_df
