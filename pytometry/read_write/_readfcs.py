from __future__ import annotations

from pathlib import Path
from typing import Any

import readfcs
from anndata import AnnData
from tqdm.auto import tqdm


def read_fcs(path: str, reindex: bool = True) -> AnnData:
    """Read FCS file and convert into AnnData format.

    Args:
        path: str or Path
            location of fcs file to parse
        reindex: boolean
            use the marker info to reindex variable names
            defaults to True

    Returns:
        an AnnData object of the fcs file
    """
    return readfcs.read(path, reindex=reindex)


def read_and_merge(
    files: str | list[str],
    sample_ids: list[Any] | None = None,
    sample_id_from_filename: bool = False,
    sample_id_index: int = 0,
    sample_id_sep: str = "_",
) -> AnnData:
    """Read and merge multiple FCS files into a single AnnData object.

    Args:
        files (str | list[str]): either a list of file paths or a directory path
        sample_ids (list[Any] | None): list of sample ids to use as a column in
        the AnnData object
        sample_id_from_filename (bool): whether to use the filename to extract the
        sample id
        sample_id_index (int): which index of the filename to use as the sample id,
        defaults to 0
        sample_id_sep (str): separator to use when splitting the filename, defaults
        to "_"

    Returns:
        AnnData: merged AnnData object
    """
    if isinstance(files, str):
        if Path(files).is_dir():
            files = [str(f) for f in Path(files).glob("*.fcs")]
        else:
            raise ValueError("files must be a list of files or a directory path")
    elif isinstance(files, list):
        files = [str(Path(f)) for f in files if Path(f).suffix == ".fcs"]

    if sample_ids is None and sample_id_from_filename:
        sample_ids = [f.split(sample_id_sep)[sample_id_index] for f in files]
    else:
        sample_ids = [None for _ in range(len(files))]

    adata_stack = []
    for file, file_id in tqdm(
        zip(files, sample_ids), total=len(files), desc="Loading FCS files"
    ):
        adata = read_fcs(file)
        if file_id is not None:
            adata.obs["sample"] = file_id
        adata_stack.append(adata)
    return AnnData.concatenate(*adata_stack, join="outer", uns_merge="unique")
