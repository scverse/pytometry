from pathlib import PosixPath

import readfcs


def read_fcs(path: str):
    """Read FCS file and convert into anndata format.

    Args:
        path (str): path or Path
            location of fcs file to parse

    Returns:
        adata: AnnData object of the fcs file
    """
    if isinstance(path, PosixPath):
        path = path.as_posix()

    fcsfile = readfcs.FCSFile(path)
    adata = fcsfile.to_anndata()

    # move marker name to index. Then merging data becomes easier
    if "marker" in adata.var_keys():
        adata.var["channel"] = adata.var_names.values
        adata.var_names = adata.var["marker"]
        del adata.var["marker"]

    return adata
