"""
Author:     Thomas Ryborz
ICB         HelmholtzZentrum münchen
Date:       15.01.2020

Fileconverter for .fcs -> .h5ad and .h5ad -> .fcs
"""


import getpass
import os.path
from pathlib import Path

import FlowCytometryTools as fct
import anndata as ann
import numpy as np
import pandas as pd

from ..preprocessing import _process_data
from . import fcswriter



def __toanndata(filenamefcs, 
                fcsfile, 
                spillover_key ='$SPILLOVER', 
                save = False):
    """
    Converts .fcs file to .h5ad file.
    :param filenamefcs: filename without extension
    :param fcsfile: path to .fcs file
    :param spillover_key: name to access the spillover matrix, if any
    :return: Anndata object with additional .uns entries
    """
    fcsdata = fct.FCMeasurement(ID='FCS-file', datafile=fcsfile)
    adata = ann.AnnData(X=fcsdata.data[:].values)
    adata.var_names = fcsdata.channel_names
    adata.uns['meta'] = fcsdata.meta

    #check for any binary format types in the .uns['meta'] dictionary
    #and replace by a string
    keys_all = adata.uns['meta'].keys()
    for key in keys_all:
        types = type(adata.uns['meta'][key])
        if types is dict:
            keys_sub = adata.uns['meta'][key].keys()
            for key2 in keys_sub:
                types2 = type(adata.uns['meta'][key][key2])
                if types2 is bytes:
                    adata.uns['meta'][key][key2] = adata.uns['meta'][key][key2].decode()
        elif types is bytes:
            adata.uns['meta'][key] = adata.uns['meta'][key].decode()
        elif types is tuple:
            adata.uns['meta'][key] = list(adata.uns['meta'][key])
        #check for data frame
        elif isinstance(adata.uns['meta'][key], pd.DataFrame):    
            dict_tmp = {}
            df_tmp = adata.uns['meta'][key]
            for col in df_tmp.columns:
                if type(df_tmp[col].iloc[0]) is list:
                #iterate over list entries
                    for n in range(len(df_tmp[col].iloc[0])):
                        dict_tmp[col + str(n+1)] = [entry[n] for entry in df_tmp[col]]    
                else:
                    df_tmp[col] = df_tmp[col].fillna(str(0))
                    dict_tmp[col] = df_tmp[col].values
            adata.uns['meta'][key] = dict_tmp

    if spillover_key in fcsdata.meta:
        adata.uns['spill_mat'] = proc.create_spillover_mat(fcsdata, 
                                                           key=spillover_key)
        adata.uns['comp_mat'] = proc.create_comp_mat(adata.uns['spill_mat'])

    if save:
        adata.write_h5ad(Path(filenamefcs + '_converted' + '.h5ad'))

    return adata


def __tofcs(filenameh5ad, anndatafile, save):
    """
    Converts .h5ad file to .fcs file.
    :param filenameh5ad: filename without extension
    :param anndatafile: path to .h5ad file
    :return: Metadata of the created .fcs file.
    """
    # String to avoid duplicate keywords
    clear_dupl = ['__header__', '_channels_', '_channel_names_',
                  '$BEGINANALYSIS', '$ENDANALYSIS', '$BEGINSTEXT', '$ENDSTEXT',
                  '$BEGINDATA', '$ENDDATA', '$BYTEORD', '$DATATYPE',
                  '$MODE', '$NEXTDATA', '$TOT', '$PAR', '$fcswrite version']

    adata = ann.read_h5ad(anndatafile)
    dictionary = adata.uns['meta']
    ch_shortnames = dictionary['_channels_'][:, 0]

    # Include long channel names in seperate Key
    count = 1

    for name in dictionary['_channel_names_']:
        dictionary['$P' + str(count) + 'S'] = name
        count = count + 1

    for i in clear_dupl:
        dictionary.pop(i, None)

    if save:
        fcswriter.write_fcs(Path(filenameh5ad + '_converted' + '.fcs'), ch_shortnames, np.array(adata.var_names).tolist(),
                            adata.X, dictionary, 'big', False)

    return fct.FCMeasurement('FCS-file', filenameh5ad + '_converted' + '.fcs')


def readandconvert(datafile='',
                   spillover_key = '$SPILLOVER',
                   save_flag=False):
    """
    Loads files and converts them according to their extension.
    :rtype: A list of loaded files.
    """
    elementlist = []

    # Path to file
    if datafile != '':
        file_names = [datafile]
    else:
        from tkinter import Tk
        from tkinter import filedialog
        username = getpass.getuser()  # current username

        file_dialog = Tk()
        file_dialog.withdraw()

        file_names = filedialog.askopenfilenames(initialdir="/home/%s/" % username, 
                                                 title="Select file",
                                                 filetypes=(("all files", "*.*"), 
                                                            ("fcs files", "*.fcs"),
                                                            ("h5ad files", ".h5ad")))

    for file_name in file_names:

        file_path = file_name
        filename, file_extension = os.path.splitext(file_path)

        if file_extension in ['.fcs', '.FCS']:
            elementlist.append(__toanndata(filename, file_path, 
                                           spillover_key,  save_flag))
        elif file_extension in ['.h5ad', '.H5AD']:
            elementlist.append(__tofcs(filename, file_path, save_flag))
        else:
            print('File ' + file_name + ' can not be converted!')
    
    if len(elementlist) == 1:
        return(elementlist[0])
    else:     
        return elementlist
