import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import rcParams
#import getpass
#import os.path

#import FlowCytometryTools as fct
from anndata import AnnData
import math

from ..tools import normalize_arcsinh

def create_spillover_mat(fcsdata, 
                         key = '$SPILLOVER'):
    """Create spillover matrix from meta data of an .fcs file.

    Args:
        fcsdata (dict): Meta data from .fcs file.
        key (str, optional): Spillover matrix as panda dataframe. 
            Defaults to '$SPILLOVER'.

    Returns:
        pd.DataFrame: Spillover matrix as pd.DataFrame
    """    
    spillover = fcsdata.meta[key].split(",")
    num_col = int(spillover[0])
    channel_names = spillover[1:(int(spillover[0]) + 1)]
    channel_data = fcsdata.meta['_channels_']

    if '$PnS' in channel_data:
        channel_renames = [str(channel_data['$PnS'][channel_data['$PnN'] == name][0]) 
                           for name in channel_names]
    else:
        channel_renames = channel_names

    spill_values = np.reshape([float(inp) 
                               for inp in spillover[(int(spillover[0]) + 1):]], 
                              [num_col, num_col])
    spill_df = pd.DataFrame(spill_values, columns=channel_renames)
    return spill_df


def create_comp_mat(spillmat, relevant_data=''):
    """Creates a compensation matrix from a spillover matrix.

    Args:
        spillmat (pd.DataFrame): Spillover matrix as pandas dataframe.
        relevant_data (str, optional):A list of channels for customized selection.
            Defaults to ''.

    Returns:
        pd.DataFrame: Compensation matrix as pandas dataframe.
    """    
    if relevant_data == '':
        comp_mat = np.linalg.inv(spillmat)
        compens = pd.DataFrame(comp_mat, columns=list(spillmat.columns))
    else:
        comp_mat = np.linalg.inv(spillmat)
        compens = pd.DataFrame(comp_mat, columns=relevant_data)

    return compens


def find_indexes(adata: AnnData, 
        key_added = 'signal_type', 
        data_type='facs'):
    """Find channels of interest for computing compensation.

    Args:
        adata (AnnData): anndata object
        key_added (str, optional): key where result vector is added to the adata.var. 
            Defaults to 'signal_type'.
        data_type (str, optional): either 'facs' or 'cytof'. 
            Defaults to 'facs'.

    Returns:
        AnnData: anndata object with a categorical vector in 
            adata.var[f'{key_added}'] 
    """        
    index = adata.var.index
    index_array = []
    
    if data_type == 'facs':

        for item in index:
            if item.endswith('-A') and not item.count('SC-'):
                index_array.append('area')
            elif item.endswith('-H') and not item.count('SC-'):
                index_array.append('height')
            else:
                index_array.append('other')
                
    elif data_type in ['cytof', 'cyTOF']:
        for item in index:
            if item.endswith('Di') or item.endswith('Dd'):
                index_array.append('element')
            else:
                index_array.append('other')
    else:
        print(f"{data_type} not recognized. Must be either 'facs' or \
               'cytof'/'cyTOF'")
    adata.var['signal_type'] = pd.Categorical(index_array)
    return adata

#rename compute bleedthr to compensate
def compensate(adata: AnnData, key = 'signal_type'):
    """Computes bleedthrough for data channels.

    Args:
        adata (AnnData): AnnData object
        key (str, optional): key where result vector is added 
            to the adata.var. Defaults to 'signal_type'.

    Returns:
        AnnData: AnnData object
    """    
    key_in = key
    compens = adata.uns['comp_mat']
    # save original data as layer
    if 'original' not in adata.layers:
        adata.layers['original'] = adata.X

    # Ignore channels 'FSC-H', 'FSC-A', 'SSC-H', 'SSC-A', 
    # 'FSC-Width', 'Time'
    if key_in not in adata.var_keys():
        adata = find_indexes(adata, data_type='facs')
    #select non other indices
    indexes = np.invert(adata.var[key_in] == 'other')
    
    bleedthrough = np.dot(adata.X[:, indexes], 
                          compens)
    adata.X[:, indexes] = bleedthrough
    return adata


def split_area(adata: AnnData, 
        key='signal_type', option='area', data_type='facs'):
    """Method to filter out height or area data.

    Args:
        adata (AnnData): AnnData object containing data.
        key (str, optional): key for adata.var where the variable type is stored. 
            Defaults to 'signal_type'.
        option (str, optional):  for choosing 'area' or 'height' in case of FACS data
            and 'element' for cyTOF data. Defaults to 'area'.
        data_type (str, optional): either 'facs' or 'cytof'/'cyTOF'. Defaults to 'facs'.

    Returns:
        AnnData: AnnData object containing area or height data
    """        
    option_key = option
    key_in = key
    
    possible_options = ['area', 'height', 'other', 'element']
    
    if option_key not in possible_options:
        print(f"{option_key} is not a valid category. Return all.")
        return adata
    #Check if indices for area and height have been computed
    if key_in not in adata.var_keys():
        adata = find_indexes(adata, data_type = data_type)
    
    index = adata.var[key_in] == option_key
    #if none of the indices is true, abort
    if sum(index)<1:
        print(f"{option_key} is not in adata.var[{key_in}]. Return all.")
        return adata
    
    non_idx = np.flatnonzero(np.invert(index))

    #merge non-idx entries in data matrix with obs
    non_cols = adata.var_names[non_idx].values
    for idx, colname in enumerate(non_cols):
        adata.obs[colname] = adata.X[:,non_idx[idx]].copy()
    
    #create new anndata object (note: removes potential objects like obsm)
    adataN = AnnData(X = adata.X[:, np.flatnonzero(index)], 
                     obs = adata.obs, 
                     var = adata.var.iloc[np.flatnonzero(index)],      
                     uns = adata.uns)
    adataN.var_names = adata.var_names[index].values 
    return adataN

# TODO: move function to plotting module
# Plot data. Choose between Area, Height both(default)
def plotdata(adata: AnnData, 
             key = 'signal_type', 
             normalize = True,
             cofactor = 10, 
             figsize = (15, 6),
             option='',
             save = '',
             **kwargs
             ):
    """Creating histogram plot from Anndata object.
    
    :param adata: AnnData object containing data.
    :param cofactor: float value to normalize with in arcsinh-transform
    :param option: Switch to choose directly between area and height data.
    :param save: Filename to save the shown figure 
    :param kwargs: Passed to :func:`matplotlib.pyplot.savefig`
    """
    option_key = option
    key_in = key
    adata_ = adata.copy()
    
    #Check if indices for area and height have been computed
    if key_in not in adata_.var_keys():
        adata_ = find_indexes(adata_)
    
    if normalize:
        adata_ = normalize_arcsinh(adata_, cofactor)
    
    if option_key not in ['area', 'height', 'other']:
        print(f"Option {option_key} is not a valid category. Return all.")
        datax = adata_.X
        var_names = adata_.var_names.values
    else:
        index = adata_.var[key_in] == option_key
        datax = adata_.X[:, index]
        var_names = adata_.var_names[index].values 
    
    if len(var_names)== 0:
        print(f"Option {option_key} led to the selection of 0 variables.\
               Nothing to plot.")
        return
        

    rcParams['figure.figsize'] = figsize

    names = var_names
    number = len(names)

    columns = 3
    rows = math.ceil(number / columns)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.6)

    for idx in range(number):
        ax = fig.add_subplot(rows, columns, idx + 1)
        sb.distplot(datax[:, names == names[idx]],
                    kde=False, norm_hist=False, 
                    bins=400, ax=ax, 
                    axlabel=names[idx])
    if save !='':
        plt.savefig(save, bbox_inches = 'tight', **kwargs)
    plt.show()
    
    return




