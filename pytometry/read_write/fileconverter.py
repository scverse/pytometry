import getpass
import os.path
from pathlib import Path

import FlowCytometryTools as fct
import pandas as pd
from anndata import AnnData

from ..preprocessing import _process_data


def __toanndata(filenamefcs, fcsfile, spillover_key="$SPILLOVER", save=False):
    """Converts .fcs file to .h5ad file.

    :param filenamefcs: filename without extension
    :param fcsfile: path to .fcs file
    :param spillover_key: name to access the spillover matrix, if any
    :return: Anndata object with additional .uns entries
    """
    fcsdata = fct.FCMeasurement(ID="FCS-file", datafile=fcsfile)
    adata = AnnData(X=fcsdata.data[:].values)
    adata.var_names = fcsdata.channel_names
    adata.uns["meta"] = fcsdata.meta

    # check for any binary format types in the .uns['meta'] dictionary
    # and replace by a string
    keys_all = adata.uns["meta"].keys()
    for key in keys_all:
        types = type(adata.uns["meta"][key])
        if types is dict:
            keys_sub = adata.uns["meta"][key].keys()
            for key2 in keys_sub:
                types2 = type(adata.uns["meta"][key][key2])
                if types2 is bytes:
                    adata.uns["meta"][key][key2] = adata.uns["meta"][key][key2].decode()
        elif types is bytes:
            adata.uns["meta"][key] = adata.uns["meta"][key].decode()
        elif types is tuple:
            adata.uns["meta"][key] = list(adata.uns["meta"][key])
        # check for data frame
        elif isinstance(adata.uns["meta"][key], pd.DataFrame):
            dict_tmp = {}
            df_tmp = adata.uns["meta"][key]
            for col in df_tmp.columns:
                if type(df_tmp[col].iloc[0]) is list:
                    # iterate over list entries
                    for n in range(len(df_tmp[col].iloc[0])):
                        dict_tmp[col + str(n + 1)] = [entry[n] for entry in df_tmp[col]]
                else:
                    df_tmp[col] = df_tmp[col].fillna(str(0))
                    dict_tmp[col] = df_tmp[col].values
            adata.uns["meta"][key] = dict_tmp

    if spillover_key in fcsdata.meta:
        adata.uns["spill_mat"] = _process_data.create_spillover_mat(
            fcsdata, key=spillover_key
        )
        adata.uns["comp_mat"] = _process_data.create_comp_mat(adata.uns["spill_mat"])

    if save:
        adata.write_h5ad(Path(filenamefcs + "_converted" + ".h5ad"))

    return adata


def readandconvert(datafile="", spillover_key="$SPILLOVER", save_flag=False):
    """Load files and converts them according to their extension.

    :rtype: A list of loaded files.
    """
    elementlist = []

    # Path to file
    if datafile != "":
        file_names = [datafile]
    else:
        from tkinter import Tk, filedialog

        username = getpass.getuser()  # current username

        file_dialog = Tk()
        file_dialog.withdraw()

        file_names = filedialog.askopenfilenames(
            initialdir="/home/%s/" % username,
            title="Select file",
            filetypes=(
                ("all files", "*.*"),
                ("fcs files", "*.fcs"),
                ("h5ad files", ".h5ad"),
            ),
        )

    for file_name in file_names:
        file_path = file_name
        filename, file_extension = os.path.splitext(file_path)

        if file_extension in [".fcs", ".FCS"]:
            elementlist.append(
                __toanndata(filename, file_path, spillover_key, save_flag)
            )
        else:
            print("File " + file_name + " can not be converted!")

    if len(elementlist) == 1:
        return elementlist[0]
    else:
        return elementlist
