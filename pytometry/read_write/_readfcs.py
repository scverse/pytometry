import readfcs
from anndata import AnnData


def read_fcs(path: str) -> AnnData:
    """Read FCS file and convert into AnnData format.

    Args:
        path: str or Path
            location of fcs file to parse

    Returns:
        an AnnData object of the fcs file
    """
    return readfcs.read(path)
