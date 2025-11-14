from pathlib import Path
from typing import Any

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
        list of sample ids to use as a column in the AnnData object
    sample_id_from_filename
        whether to use the filename to extract the sample id
    sample_id_index
        which index of the filename to use as the sample id
    sample_id_sep
        separator to use when splitting the filename
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
    Merged AnnData object
    """
    if isinstance(files, str):
        if not Path(files).is_dir():
            raise ValueError("files must be a list of files or a directory path")
        files = [str(f) for f in Path(files).glob("*.fcs")]
    elif isinstance(files, list):
        files = [str(Path(f)) for f in files if Path(f).suffix == ".fcs"]

    if sample_ids is None and sample_id_from_filename:
        sample_ids = [f.split(sample_id_sep)[sample_id_index] for f in files]
    else:
        sample_ids = [None for _ in range(len(files))]

    adata_stack = []
    for file, file_id in tqdm(zip(files, sample_ids, strict=False), total=len(files), desc="Loading FCS files"):
        adata = read_fcs(
            file,
            reindex=reindex,
            ignore_offset_error=ignore_offset_error,
            ignore_offset_discrepancy=ignore_offset_discrepancy,
            use_header_offsets=use_header_offsets,
        )
        if file_id is not None:
            adata.obs["sample"] = file_id
        adata_stack.append(adata)
    return AnnData.concatenate(*adata_stack, join="outer", uns_merge="unique")
