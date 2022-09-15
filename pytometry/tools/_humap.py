from scipy.stats import entropy
from scipy.special import softmax
# import multiprocessing as mp
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scanpy.tools._umap import umap as embed_umap
from anndata import AnnData
# import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

HELPER_VAR = dict()

class _Scale:
    """
    Class that contains information about a Scale
    Properties:
    self.X
       datapoints
    self.T
       Transition matrix
    self.I
       Area of Influence matrix
    self.W
       Weight vector
    self.P
       Probability distribution
    self.X_hsne
       Embedding positions
    self.lm_ind
       landmark indices
    self.parent_scale
       parent scale (_Scale object)


    """
    def __init__(self, X=None, T=None, I=None, W=None, P=None, X_hsne=None, lm_ind=None, parent_scale=None):
        self.X = X
        self.T = T
        self.I = I
        self.W = W
        self.P = P
        self.X_hsne = X_hsne
        self.lm_ind = lm_ind
        self.parent_scale = parent_scale

        self.drilled_scale_list = list()   # TODO remove


def humap(
        adata,
        beta=100,
        beta_thresh=1.5,
        theta=50,
        num_scales=1,
        include_root_object=False,
        verbose=False,
        copy: bool = False,
        **kwargs
):
    '''

    Parameters
    ----------
    adata
       anndata object
    beta
       beta for landmark search
    beta_thresh
       beta_thresh for landmark search
    theta
       theta for Area of Influence calculation
    num_scales
       number of scales, that will be calculated
    include_root_object
       boolean value: true if a simple tSNE should be conducted on the data (first layer)
    verbose
       verbose true or false
    copy
        Return a copy instead of writing to adata
    kwargs
        additional parameters supplied to the sc.tl.umap

    Returns
       Depending on `copy`, returns or updates `adata` with the following fields.
       **hsne_scales** : `adata.uns` field
            List of HSNE Scales
    -------
    Usage:
        Given an anndata object "adata"
        k-nearest-neighbor graph required
        (=> scanpy.pp.neighbors(adata))
        then call
         => tl.hsne(adata)
        adata now contains the calculated scales (in .uns['hsne_scales'])
    '''

    adata = adata.copy() if copy else adata

    # settings dict for all important setting variables
    settings = {
        'beta': beta,
        'beta_thresh': beta_thresh,
        'theta': theta
    }

    # manage k nearest neighbors
    try:
        adata.obsp['distances']
    except KeyError:
        raise Exception("k-nearest-neighbor graph has to be constructed first")
    distances_nn = adata.obsp['distances']

    if verbose: print('Starting humap: %d points and %d scales'%(adata.X.shape[0],num_scales))
    scale_list = list()

    # Create first scale
    s_root = _Scale(X=adata.X, W=1)

    if verbose: print('T')
    s_root.T = _calc_first_T(distances_nn, adata.X.shape[0])
    if verbose: print('P')
    s_root.P = _calc_P(s_root.T)
    if include_root_object:
        if verbose: print('X_humap')
        # Construct a temporary adata to store weighted adjacency matrix
        tmpdata = AnnData(X=s_root.X)
        tmpdata.uns['neighbors'] = {'params': {'method': 'umap'},
                                    'connectivities_key': 'connectivities'
                                    }
        tmpdata.obsp['connectivities'] = s_root.P
        # sc.tl.umap uses data matrix X or X_pca just to identify the number of connected components
        # during initialization of umap embedding
        tmpdata.obsm['X_pca'] = s_root.X  # Set X_pca to X to supress warning and prevent calculation of PCA.
        embed_umap(tmpdata, **kwargs)
        s_root.X_humap = tmpdata.obsm['X_umap']

    if verbose: print('lm_ind')
    s_root.lm_ind = _get_landmarks(s_root.T, settings)

    scale_list.append(s_root)   # appending scale

    for i in range(num_scales):
        if verbose: print('Scale Number %d with %d points:' %(i,len(scale_list[i].lm_ind)))
        s_prev = scale_list[i]
        s_curr = _Scale(X=s_prev.X[s_prev.lm_ind, :], parent_scale=s_prev)
        if verbose: print('I')
        s_curr.I = _calc_AoI(s_prev)
        if verbose: print('W')
        s_curr.W = _calc_Weights(s_curr.I, s_prev.W)
        if verbose: print('T')
        s_curr.T = _calc_next_T(s_curr.I, s_prev.W)
        if verbose: print('lm_ind')
        s_curr.lm_ind = _get_landmarks(s_curr.T, settings)
        if verbose: print('P')
        s_curr.P = _calc_P(s_curr.T)
        if verbose: print('X_humap')
        tmpdata = AnnData(X=s_curr.X)
        tmpdata.uns['neighbors'] = {'params': {'method': 'umap'},
                                    'connectivities_key': 'connectivities'
                                    }
        tmpdata.obsp['connectivities'] = s_curr.P
        tmpdata.obsm['X_pca'] = s_curr.X
        embed_umap(tmpdata, **kwargs)
        s_curr.X_humap = tmpdata.obsm['X_umap']
        scale_list.append(s_curr)

    if include_root_object == False:
        scale_list.pop(0)

    adata.uns['humap_settings'] = settings
    adata.uns['humap_scales'] = scale_list
    return adata if copy else None


def _calc_P(T):
    '''

    Parameters
    ----------
    T
       transition matrix

    Returns
       joint probabilities matrix P
    -------

    '''
    return T + T.transpose() - T.multiply(T.transpose())

def _calc_Weights(I, W_old):
    '''

    Parameters
    ----------
    I
       Area of Influence matrix
    W_old
       old weights vector

    Returns
       new weights vector
    -------

    '''
    if type(W_old) is int: #W_old is None or W_old is 1:
        W_old = np.ones((I.shape[0],))
    W_s = np.array(W_old * I).reshape((I.shape[1]))
    return W_s

def _calc_next_T(I, W):
    '''

    Parameters
    ----------
    I
       Area of Influence matrix
    W
       weight vector

    Returns
       next transition matrix T
    -------

    '''
    num_lm_s_prev, num_lm_s = (I.shape[0],I.shape[1])  # dimensions of I
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
    ### Multiprocessing helper method ###
    # load globals
    W = HELPER_VAR['W']
    num_lm_s_prev = HELPER_VAR['num_lm_s_prev']
    return csr_matrix(np.reshape(i.toarray().reshape((num_lm_s_prev,)) * W, (num_lm_s_prev, 1)))

def _helper_method_T_next_row_div(r):
    ### Multiprocessing helper method ###
    return r[1] / np.sum(r[1])

def _calc_AoI(scale, min_lm=100):
    '''
    Calculates the Area of Influence of a given scale

    '''
    n_events = scale.T.shape[0]
    # create state matrix containing all initial states
    init_states = csr_matrix((np.ones(n_events), (range(n_events), range(n_events))))
    # save temp static variables in global for outsourced multiprocessing method
    global HELPER_VAR
    HELPER_VAR = {'lm': scale.lm_ind, 'min_lm': min_lm, 'T': scale.T}
    I = map(_helper_method_AoI, [s for s in init_states])
    I = vstack(I)
    return I

def _helper_method_AoI(state):
    ### Multiprocessing helper method ###
    # load globals
    T = HELPER_VAR['T']
    lm = HELPER_VAR['lm']
    reached_lm = np.zeros(len(lm))

    cache = state
    state_len = np.shape(state)[1]  # get length of vector once

    # do until minimal landmark-"hit"-count is reached (--> landmarks_left < 0)
    landmarks_left = HELPER_VAR['min_lm']
    while landmarks_left >= 0:
        # erg_random_walk = -1
        step = 1
        while True:
            cache = cache * T
            erg_random_walk = np.random.choice(state_len, p=cache.toarray()[0])
            if erg_random_walk in lm:
                reached_lm[lm.index(erg_random_walk)] += 1
                landmarks_left -= 1
                break
            step += 1
    erg = reached_lm / np.sum(reached_lm.data)
    return csr_matrix(erg)

def _get_landmarks(T, settings):
    '''
    Parameters
    ----------
    T
       Transition matrix
    settings
       settings dict

    Returns
       list of landmark indices
    -------
    '''
    n_events = T.shape[0]
    proposals = np.zeros(n_events)  # counts how many times point has been reached
    landmarks = list()  # list of landmarks
    global HELPER_VAR
    HELPER_VAR = {'T': T,
                  'theta': settings['theta'],
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
    ### Multiprocessing helper method ###
    for i in range(HELPER_VAR['theta']):
        state *= HELPER_VAR['T']
    destinations = np.random.choice(range(HELPER_VAR['n_events']), HELPER_VAR['beta'], p=state.toarray()[0])
    hits = np.zeros((HELPER_VAR['n_events']))
    for d in destinations:
        hits[d] += 1
    return [(h[0], h[1]) for h in enumerate(hits) if h[1] > 0]

def _calc_first_T(distances_nn, dim):
    '''
    Parameters
    ----------
    distances_nn
       distances of nearest neighbor graph
    dim
       dimensions of Transition matrix

    Returns
       Transition matrix
    -------
    '''
    probs = map(_helper_method_calc_T, [dist.data for dist in distances_nn])

    data = []
    for pr in probs:
        data.extend(pr)
    T = csr_matrix((data, distances_nn.indices, distances_nn.indptr), shape=(dim,dim))
    return T


def _helper_method_calc_T(dist):
    ### Multiprocessing helmper method ###
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


############################################################################
### Lasso Selector                                                       ###
### Not finished yet. Used to drill into Data                            ###
############################################################################

class _SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.
    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.
    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).
    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.
    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
