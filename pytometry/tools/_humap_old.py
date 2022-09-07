from scipy.stats import entropy
from scipy.special import softmax
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack, hstack
from scanpy.tools._umap import umap as embed_umap
from anndata import AnnData
HELPER_VAR = dict()


class _Scale:
    def __init__(self, X=None, T=None, I=None, W=None, P=None, X_humap=None, lm_idx=None, parent_scale=None):
        self.X = X
        self.T = T
        self.I = I
        self.W = W
        self.P = P
        self.X_humap = X_humap
        self.lm_idx = lm_idx
        self.parent_scale = parent_scale


def humap(adata, channel_idx=None, beta=100, beta_thresh=1.5, teta=50, n_scales=1, copy=False):
    if channel_idx is None:
        channel_idx = range(len(adata.var_names))
    elif len(channel_idx) == 0:
        channel_idx = range(len(adata.var_names))

    # settings dict for all important setting variables
    settings = {
        'beta': beta,
        'beta_thresh': beta_thresh,
        'teta': teta,
        'channel_idx': channel_idx
    }

    # manage k nearest neighbors
    try:
        adata.obsp['distances']
    except KeyError as e:
        raise Exception("k-nearest-neighbor graph has to be constructed first")
    distances_nn = adata.obsp['distances']

    scale_list = list()

    # Create first scale
    s_root = _Scale(X=adata.X[:,channel_idx], W=1)  # reduced x to channels_idx


    s_root.T = _calc_first_T(distances_nn, len(adata.X))
    s_root.P = _calc_P(s_root.T)
    # Construct a temporary adata to store weighted adjacency matrix
    tmpdata = AnnData(X=s_root.X)
    tmpdata.uns['neighbors'] = {'params': {'method': 'umap'},
                                'connectivities_key': 'connectivities'
                                }
    tmpdata.obsp['connectivities'] = s_root.P
    # sc.tl.umap uses data matrix X or X_pca just to identify the number of connected components
    # during initialization of umap embedding
    tmpdata.obsm['X_pca'] = s_root.X # Set X_pca to X to supress warning and prevent calculation of PCA.

    embed_umap(tmpdata)
    s_root.X_humap = tmpdata.obsm['X_umap']
    s_root.lm_idx = _get_landmarks(s_root.T, settings)

    scale_list.append(s_root)

    for i in range(n_scales):
        print(f'Scale Number {i}')
        s_prev = scale_list[i]
        s_curr = _Scale(X=s_prev.X[s_prev.lm_idx, :], parent_scale=s_prev)
        s_curr.I = _calc_AoI(s_prev)
        s_curr.W = _calc_Weights(s_curr.I, s_prev.W)
        s_curr.T = _calc_next_T(s_curr.I, s_prev.W)
        s_curr.lm_idx = _get_landmarks(s_curr.T, settings)
        s_curr.P = _calc_P(s_curr.T)

        tmpdata = AnnData(X=s_curr.X)
        tmpdata.uns['neighbors'] = {'params': {'method': 'umap'},
                                    'connectivities_key': 'connectivities'
                                    }
        tmpdata.obsp['connectivities'] = s_curr.P
        tmpdata.obsm['X_pca'] = s_curr.X
        embed_umap(tmpdata)
        s_curr.X_humap = tmpdata.obsm['X_umap']
        scale_list.append(s_curr)

    adata.uns['humap_settings'] = settings
    adata.uns['humap_scales'] = scale_list
    return adata if copy else None


## Helper functions
def _calc_P(T):
    P = T + T.transpose() - T.multiply(T.transpose())
    return P


def _calc_Weights(I, W_old):
    if type(W_old) is int:  # W_old is None or W_old is 1:
        W_old = np.ones((I.shape[0],))
    W_s = np.array(W_old * I).reshape((I.shape[1]))
    return W_s


def _calc_next_T(I, W):
    num_lm_s_prev, num_lm_s = (I.shape[0],I.shape[1])  # dimensionst of I
    # num_lm_s_old > num_lm_s

    I_t = I.transpose()  # transposed Influence matrix

    global HELPER_VAR
    HELPER_VAR = {'W': W, 'num_lm_s_prev': num_lm_s_prev}

    I_with_W = map(_helper_method_T_next_mul_W, [it for it in I_t])
    I_with_W = hstack(list(I_with_W))
    I = I_with_W.T * I
    T_next = map(_helper_method_T_next_row_div, enumerate(I))

    T_next = vstack(T_next)
    return T_next.tocsr()


def _helper_method_T_next_mul_W(i):
    # load globals
    W = HELPER_VAR['W']
    num_lm_s_prev = HELPER_VAR['num_lm_s_prev']
    return csr_matrix(np.reshape(i.toarray().reshape((num_lm_s_prev,)) * W, (num_lm_s_prev, 1)))


def _helper_method_T_next_row_div(r):
    return r[1] / np.sum(r[1])


def _calc_AoI(scale, min_lm=100):
    n_events = scale.T.shape[0]
    # create state matrix containing all initial states
    init_states = csr_matrix((np.ones(n_events), (range(n_events), range(n_events))))
    # save temp static variables in global for outsourced multiprocessing method
    global HELPER_VAR
    HELPER_VAR = {'lm': scale.lm_idx, 'min_lm': min_lm, 'T': scale.T}

    I = map(_helper_method_AoI, [s for s in init_states])
    I = vstack(I)
    return I


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


def _get_landmarks(T, settings):

    n_events = T.shape[0]
    proposals = np.zeros(n_events)  # counts how many times point has been reached
    landmarks = list()  # list of landmarks
    global HELPER_VAR
    HELPER_VAR = {'T': T,
                  'teta': settings['teta'],
                  'beta': settings['beta'],
                  'beta_thresh': settings['beta_thresh'],
                  'n_events': n_events}
    init_states = csr_matrix((np.ones(n_events), (range(n_events), range(n_events))))
    hit_list = map(_helper_method_get_landmarks, [state for state in init_states])
    # evaluate results
    for state_hits in hit_list:  # for every states hit_list
        for h in state_hits:  # for every hit in some states hit_list
            proposals[h[0]] += h[1]

    # collect landmarks
    min_beta = settings['beta'] * settings['beta_thresh']
    for prop in enumerate(proposals):
        # if event has been hit min_beta times, it counts as landmark
        if prop[1] > min_beta:
            landmarks.append(prop[0])
    return landmarks


def _helper_method_get_landmarks(state):
    for i in range(HELPER_VAR['teta']):
        state *= HELPER_VAR['T']
    destinations = np.random.choice(range(HELPER_VAR['n_events']), HELPER_VAR['beta'], p=state.toarray()[0])
    hits = np.zeros((HELPER_VAR['n_events']))
    for d in destinations:
        hits[d] += 1
    return [(h[0], h[1]) for h in enumerate(hits) if h[1] > 0]


def _calc_first_T(distances_nn, dim):
    probs = map(_helper_method_calc_T, [dist.data for dist in distances_nn])
    data = []
    for pr in probs:
        data.extend(pr)
    T = csr_matrix((data, distances_nn.indices, distances_nn.indptr), shape=(dim, dim))
    return T


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
