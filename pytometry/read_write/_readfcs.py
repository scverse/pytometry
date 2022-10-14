import readfcs
from anndata import AnnData


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
