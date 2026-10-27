from pathlib import Path
from typing import Any

import anndata
import readfcs
from anndata import AnnData
from tqdm.auto import tqdm


def read_fcs(
    path: str,
    reindex: bool = True,
    ignore_offset_error: bool = False,
    ignore_offset_discrepancy: bool = False,
    use_header_offsets: bool = False,
) -> AnnData:
    """Read FCS file and convert into AnnData format.

    Parameters
    ----------
    path
        location of fcs file to parse
    reindex
        use the marker info to reindex variable names
    ignore_offset_error
        Ignore data offset error. Default is False
    ignore_offset_discrepancy
        Ignore discrepancy between the HEADER and TEXT values for the DATA byte offset location.
        Default is False
    use_header_offsets
        Use the HEADER section for the data offset locations. Default is False.
        Setting this option to True also suppresses an error in cases of an offset discrepancy.

    Returns
    -------
    An AnnData object of the fcs file
    """
    return readfcs.read(
        path,
        reindex=reindex,
        ignore_offset_error=ignore_offset_error,
        ignore_offset_discrepancy=ignore_offset_discrepancy,
        use_header_offsets=use_header_offsets,
    )


def read_and_merge(
    files: str | list[str],
    sample_ids: list[Any] | None = None,
    sample_id_from_filename: bool = False,
    sample_id_index: int = 0,
    sample_id_sep: str = "_",
    reindex: bool = True,
    ignore_offset_error: bool = False,
    ignore_offset_discrepancy: bool = False,
    use_header_offsets: bool = False,
) -> AnnData:
    """Read and merge multiple FCS files into a single AnnData object.

    Parameters
    ----------
    files
        either a list of file paths or a directory path
    sample_ids
        sample id of each FCS file, in the order of `files` (sorted by name for a directory), added as
        `adata.obs["sample"]`
    sample_id_from_filename
        whether to take the sample ids from the file names, if `sample_ids` is None
    sample_id_index
        which part of the file name (without its extension) to use as the sample id
    sample_id_sep
        separator to use when splitting the file name
    reindex
        use the marker info to reindex variable names
    ignore_offset_error
        Ignore data offset error. Default is False
    ignore_offset_discrepancy
        Ignore discrepancy between the HEADER and TEXT values for the DATA byte offset location.
        Default is False
    use_header_offsets
        Use the HEADER section for the data offset locations. Default is False.
        Setting this option to True also suppresses an error in cases of an offset discrepancy.

    Returns
    -------
    Merged AnnData object, with the index of each cell's file in `adata.obs["batch"]`
    """
    if isinstance(files, str):
        if not Path(files).is_dir():
            raise ValueError("files must be a list of files or a directory path")
        files = sorted(str(f) for f in Path(files).iterdir() if f.suffix.lower() == ".fcs")
    elif isinstance(files, list):
        files = [str(Path(f)) for f in files if Path(f).suffix.lower() == ".fcs"]

    if sample_ids is None and sample_id_from_filename:
        sample_ids = [Path(f).stem.split(sample_id_sep)[sample_id_index] for f in files]
    elif sample_ids is not None and len(sample_ids) != len(files):
        raise ValueError(f"Got {len(sample_ids)} sample_ids for {len(files)} FCS files")

    adata_stack = []
    for i, file in enumerate(tqdm(files, desc="Loading FCS files")):
        adata = read_fcs(
            file,
            reindex=reindex,
            ignore_offset_error=ignore_offset_error,
            ignore_offset_discrepancy=ignore_offset_discrepancy,
            use_header_offsets=use_header_offsets,
        )
        if sample_ids is not None:
            adata.obs["sample"] = sample_ids[i]
        adata_stack.append(adata)
    return anndata.concat(
        adata_stack,
        join="outer",
        merge="same",
        uns_merge="unique",
        label="batch",
        keys=[str(i) for i in range(len(adata_stack))],
        index_unique="-",
    )
