import scanpy as sc
import pytometry as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = 'G:\\My Drive\\colab\\cytometry\\2022_Nature_Becher\\surface_panel\\batch_1\\'
fls = sorted(os.listdir(BASE))
fls.pop(0)
first = fls[0]

fpath = BASE + first
adata = pt.io.read_fcs(fpath)

sc.pp.subsample(adata, 0.05)
sc.pp.neighbors(adata)

from scipy.stats import entropy
from scipy.special import softmax
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack, hstack
from scanpy.tools._umap import umap
import joblib
from typing import Union, Optional, Literal
_InitPos = Literal['paga', 'spectral', 'random']
AnyRandom = Union[None, int, np.random.RandomState]
HELPER_VAR = {}

class _Scale:
    def __init__(self, X=None, T=None, I=None, W=None, P=None, X_humap=None, lm_ind=None, parent_scale=None):
        self.X = X
        self.T = T
        self.I = I
        self.W = W
        self.P = P
        self.X_humap = X_humap
        self.lm_ind = lm_ind
        self.parent_scale = parent_scale


imp_channel_ind=None
beta=100
beta_thresh=1.5
teta=50
num_scales=1
min_dist= 0.5
spread= 1.0
n_components= 2
maxiter= None
alpha= 1.0
gamma= 1.0
negative_sample_rate= 5
init_pos='spectral'
random_state= 0
a= None
b= None
copy= False
method= 'umap'
neighbors_key= None

if imp_channel_ind is None:
    imp_channel_ind = range(len(adata.var_names))
elif len(imp_channel_ind) == 0:
    imp_channel_ind = range(len(adata.var_names))

# settings dict for all important setting variables
parameters = {
    'beta': beta,
    'beta_thresh': beta_thresh,
    'teta': teta,
    'imp_channel_ind' : imp_channel_ind}

try:
    adata.obsp['distances']
except KeyError as e:
    raise Exception("k-nearest-neighbor graph has to be constructed first")
distances_nn = adata.obsp['distances']

scale_list = list()

adata.uns['humap_settings'] = parameters
adata.uns['humap_scales'] = scale_list

def _helper_method_calc_T(dist):
    d = dist / np.max(dist)
    return softmax((-d ** 2) / _binary_search_sigma(d, len(d)))

def _binary_search_sigma(d, n_neigh):
    # binary search
    sigma = 10  # Start Sigma
    goal = np.log(n_neigh)  # log(k) with k being n_neighbors
    # Do binary search until entropy ~== log(k)
    while True:
        ent = entropy(softmax((-d ** 2) / sigma))
        # check sigma
        if np.isclose(ent, goal):
            return sigma
        if ent > goal:
            sigma *= 0.5
        else:
            sigma /= 0.5

def _helper_method_AoI(state):
    # load globals
    T = HELPER_VAR['T']
    lm = HELPER_VAR['lm']
    reached_lm = np.zeros(len(lm))

    cache = list()  # create empty cache list
    cache.append(state)  # append initial state vector as first element
    state_len = np.shape(state)[1]  # get length of vector once

    # do until minimal landmark-"hit"-count is reached (--> landmarks_left < 0)
    landmarks_left = HELPER_VAR['min_lm']
    while landmarks_left >= 0:
        # erg_random_walk = -1
        step = 1
        while True:
            if len(cache) <= step:
                cache.append(cache[step - 1] * T)
            erg_random_walk = np.random.choice(state_len, p=cache[step].toarray()[0])
            if erg_random_walk in lm:
                reached_lm[lm.index(erg_random_walk)] += 1
                landmarks_left -= 1
                break
            step += 1
    erg = reached_lm / np.sum(reached_lm.data)
    return csr_matrix(erg)

def _helper_method_T_next_mul_W(i):
    # load globals
    W = HELPER_VAR['W']
    num_lm_s_prev = HELPER_VAR['num_lm_s_prev']
    return csr_matrix(np.reshape(i.toarray().reshape((num_lm_s_prev,)) * W, (num_lm_s_prev, 1)))

def _helper_method_T_next_row_div(r):
    return r[1] / np.sum(r[1])

def _helper_method_get_landmarks(state):
    for i in range(HELPER_VAR['teta']):
        state *= HELPER_VAR['T']
    destinations = np.random.choice(range(HELPER_VAR['n_events']), HELPER_VAR['beta'], p=state.toarray()[0])
    hits = np.zeros((HELPER_VAR['n_events']))
    for d in destinations:
        hits[d] += 1
    return [(h[0], h[1]) for h in enumerate(hits) if h[1] > 0]

import time
from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocess import Pool
if __name__ == '__main__':
    print('name is eq to main')
    t1 = time.time()
    p = Pool(8)
    probs = p.map(_helper_method_calc_T, [dist.data for dist in distances_nn])
    p.terminate()
    p.join()
    t2 = time.time()
    print(f'exec time {t2 - t1}')

if __name__ == '__main__':
    print('Using built-in mp')
    t1 = time.time()
    p = mp.Pool(8)
    probs = p.map(_helper_method_calc_T, [dist.data for dist in distances_nn])
    p.terminate()
    p.join()
    t2 = time.time()
    print(f'exec time {t2 - t1}')

if __name__ == '__main__':
    print('Using bare map')
    t1 = time.time()
    probs = map(_helper_method_calc_T, [dist.data for dist in distances_nn])
    t2 = time.time()
    print(f'exec time {t2 - t1}')

# Error ia prevented