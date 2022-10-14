import numpy as np
from anndata import AnnData
from scipy import interpolate


def normalize_arcsinh(adata: AnnData, cofactor: float, inplace: bool = True):
    """Inverse hyperbolic sine transformation.

    Args:
        adata : AnnData object
        cofactor (float): all values are divided by this
           factor before arcsinh transformation recommended value for
           cyTOF data is 5 and for flow data 150.
        inplace (bool, optional): Return a copy instead of writing to adata.
            Defaults to True.

    Returns:
        Depending on `inplace`, returns or updates `adata`
        in the following field `adata.X` is then a normalised
        adata object
    """
    adata = adata if inplace else adata.copy()
    adata.X = np.arcsinh(adata.X / cofactor)
    return None if inplace else adata


def normalize_logicle(adata, t=262144, m=4.5, w=0.5, a=0, inplace=True):
    """Logicle transformation.

    Args:
        adata (AnnData): AnnData object
        t (float, optional): parameter for the top of the linear scale.
            Defaults to 262144.
        m (float, optional): parameter for the number of decades
            the true logarithmic scale approaches at the high end of
            the scale. Defaults to 4.5.
        w (float, optional): parameter for the approximate number of
            decades in the linear region. Defaults to 0.5.
        a (float, optional): parameter for the additional number of
            negative decades. Defaults to 0.
        copy (bool, optional): Return a inplace instead of writing to adata.
            Defaults to True.

    Returns:
        Depending on `inplace`, returns or updates `adata`
        in the following field `adata.X` is then a normalised
        adata object

    Details:
        Logicle transformation, implemented as defined in the
        GatingML 2.0 specification, adapted from FlowKit and Flowutils
        Python packages.

        logicle(x, T, W, M, A) = root(B(y, T, W, M, A) - x)

        where B is a modified bi-exponential function defined as

        B(y, T, W, M, A) = ae^(by) - ce^(-dy) - f

        The Logicle transformation was originally defined in the
        publication of

        Moore WA and Parks DR. Update for the logicle data scale
        including operational code implementations.
        Cytometry A., 2012:81A(4):273-277.
    """
    # initialise precision
    taylor_length = 16
    # initialise parameter dictionary
    p = dict()

    T = t
    M = m
    W = w
    A = a

    # actual parameters
    # formulas from bi-exponential paper
    p["w"] = W / (M + A)
    p["x2"] = A / (M + A)
    p["x1"] = p["x2"] + p["w"]
    p["x0"] = p["x2"] + 2 * p["w"]
    p["b"] = (M + A) * np.log(10)
    p["d"] = _solve(p["b"], p["w"])

    c_a = np.exp(p["x0"] * (p["b"] + p["d"]))
    mf_a = np.exp(p["b"] * p["x1"]) - c_a / np.exp(p["d"] * p["x1"])
    p["a"] = T / ((np.exp(p["b"]) - mf_a) - c_a / np.exp(p["d"]))
    p["c"] = c_a * p["a"]
    p["f"] = -mf_a * p["a"]

    # use Taylor series near x1, i.e., data zero to
    # avoid round off problems of formal definition
    p["xTaylor"] = p["x1"] + p["w"] / 4

    # compute coefficients of the Taylor series
    posCoef = p["a"] * np.exp(p["b"] * p["x1"])
    negCoef = -p["c"] / np.exp(p["d"] * p["x1"])

    # 16 is enough for full precision of typical scales
    p["taylor"] = np.zeros(taylor_length)

    for i in range(0, taylor_length):
        posCoef *= p["b"] / (i + 1)
        negCoef *= -p["d"] / (i + 1)
        p["taylor"][i] = posCoef + negCoef

    p["taylor"][1] = 0  # exact result of Logicle condition

    # end original initialize method
    adata = adata if inplace else adata.copy()
    # apply scaling to each value
    for i in range(0, adata.n_vars):
        for j in range(0, adata.n_obs):
            adata.X[j, i] = _scale(adata.X[j, i], p)

    return None if inplace else adata


def _scale(value: float, p: dict) -> float:
    """Scale helper function.

    Args:
        value (float): Entry in the anndata matrix
        p (dict): Parameter dictionary

    Returns:
       float: Scaled value or -1
    """
    DBL_EPSILON = 1e-9  # from C++,
    # defined as the smallest difference between 1
    # and the next larger number
    # handle true zero separately
    if value == 0:
        return p["x1"]

    # reflect negative values
    negative = value < 0
    if negative:
        value = -value

    # initial guess at solution

    if value < p["f"]:
        # use linear approximation in the quasi linear region
        x = p["x1"] + value / p["taylor"][0]
    else:
        # otherwise use ordinary logarithm
        x = np.log(value / p["a"]) / p["b"]

    # try for double precision unless in extended range
    tolerance = 3 * DBL_EPSILON
    if x > 1:
        tolerance = 3 * x * DBL_EPSILON

    for i in range(0, 40):
        # compute the function and its first two derivatives
        ae2bx = p["a"] * np.exp(p["b"] * x)
        ce2mdx = p["c"] / np.exp(p["d"] * x)

        if x < p["xTaylor"]:
            # near zero use the Taylor series
            y = _seriesBiexponential(p, x) - value
        else:
            # this formulation has better round-off behavior
            y = (ae2bx + p["f"]) - (ce2mdx + value)
        abe2bx = p["b"] * ae2bx
        cde2mdx = p["d"] * ce2mdx
        dy = abe2bx + cde2mdx
        ddy = p["b"] * abe2bx - p["d"] * cde2mdx

        # this is Halley's method with cubic convergence
        delta = y / (dy * (1 - y * ddy / (2 * dy * dy)))
        x -= delta

        # if we've reached the desired precision we're done
        if abs(delta) < tolerance:
            # handle negative arguments
            if negative:
                return 2 * p["x1"] - x
            else:
                return x

    # if we get here, scale did not converge
    return -1


def _solve(b: float, w: float) -> float:
    """Helper function for biexponential transformation.

    Args:
        b (float): parameter for biex trafo
        w (float): parameter for biex trafo
    """
    DBL_EPSILON = 1e-9  # from C++, defined as the
    # smallest difference between 1
    # and the next larger number

    # w == 0 means its really arcsinh
    if w == 0:
        return b

    # precision is the same as that of b
    tolerance = 2 * b * DBL_EPSILON

    # based on RTSAFE from Numerical Recipes 1st Edition
    # bracket the root
    d_lo = 0.0
    d_hi = b

    # bisection first step
    d = (d_lo + d_hi) / 2
    last_delta = d_hi - d_lo

    # evaluate the f(w,b) = 2 * (ln(d) - ln(b)) + w * (b + d)
    # and its derivative
    f_b = -2 * np.log(b) + w * b
    f = 2 * np.log(d) + w * d + f_b
    last_f = np.nan

    for i in range(1, 40):
        # compute the derivative
        df = 2 / d + w

        # if Newton's method would step outside the bracket
        # or if it isn't converging quickly enough
        if ((d - d_hi) * df - f) * ((d - d_lo) * df - f) >= 0 or abs(1.9 * f) > abs(
            last_delta * df
        ):
            # take a bisection step
            delta = (d_hi - d_lo) / 2
            d = d_lo + delta
            if d == d_lo:
                return d  # nothing changed, we're done
        else:
            # otherwise take a Newton's method step
            delta = f / df
            t = d
            d -= delta
            if d == t:
                return d  # nothing changed, we're done

        # if we've reached the desired precision we're done
        if abs(delta) < tolerance:
            return d
        last_delta = delta

        # recompute the function
        f = 2 * np.log(d) + w * d + f_b
        if f == 0 or f == last_f:
            return d  # found the root or are not going to get any closer
        last_f = f

        # update the bracketing interval
        if f < 0:
            d_lo = d
        else:
            d_hi = d

    return -1


def _seriesBiexponential(p: dict, value: float) -> float:
    """Helper function to compute biex trafo.

    Args:
        p (dict): Parameter dictionary
        value (float): Start value for Taylor series expansion
    """
    # initialise precision
    taylor_length = 16
    # Taylor series is around x1
    x = value - p["x1"]
    # note that taylor[1] should be identically zero according
    # to the Logicle condition so skip it here
    sum1 = p["taylor"][taylor_length - 1] * x
    for i in range(taylor_length - 2, 1, -1):
        sum1 = (sum1 + p["taylor"][i]) * x

    return (sum1 * x + p["taylor"][0]) * x


def normalize_biExp(
    adata,
    negative=0.0,
    width=-10.0,
    positive=4.418540,
    max_value=262144.000029,
    inplace=True,
):
    """Biexponential transformation.

    Biex transform as implemented in FlowJo 10. Adapted from FlowKit
    Python package. This transform is applied exactly as the FlowJo 10
    is implemented, using lookup tables with only a limited set
    of parameter values.

    Information on the input parameters from the FlowJo docs can be found in the
    details section.

    Args:
        adata: AnnData object representing the FCS data
        negative (float, optional): Value for the FlowJo biex option 'negative' (float)
            or pd.Series. Defaults to 0.0.
        width (float, optional): Value for the FlowJo biex option 'width' (float) or
            pd.Series. Defaults to -10.0.
        positive (float, optional): Value for the FlowJo biex option 'positive' (float)
            or pd.Series. Defaults to 4.418540.
        max_value (float, optional): parameter for the top of the linear scale
            or pd.Series. Defaults to 262144.000029.
        inplace (bool, optional): Return a copy instead of writing to adata.
            Defaults to True.

    Returns:
        Depending on `inplace`, returns or updates `adata` in the
        following field `adata.X` is then a normalised adata object

    Details:
        Adjusting width: The value for `w` will determine the amount of channels to be
            compressed into linear space around zero. The space of linear does
            not change, but rather the number of channels or bins being
            compressed into the linear space. Width should be set high enough
            that all of the data in the histogram is visible on screen, but not
            so high that extra white space is seen to the left hand side of your
            dimmest distribution. For most practical uses, once all events have
            been shifted off the axis and there is no more axis 'pile-up', then
            the optimal width basis value has been reached.
        Negative:
            Another component in the biexponential transform calculation is the
            negative decades or negative space. This is the only other value you
            will probably ever need to adjust. In cases where a high width basis
            may start compressing dim events into the negative cluster, you may
            want to lower the width basis (less compression around zero) and
            instead, increase the negative space by 0.5 - 1.0. Doing this will
            expand the space around zero so the dim events are still visible,
            but also expand the negative space to remove the cells from the axis
            and allow you to see the full distribution.
        Positive:
            The presence of the positive decade adjustment is due to the
            algorithm used for logicle transformation, but is not useful in
            99.9% of the cases that require adjusting the biexponential
            transform. It may be appropriate to adjust this value only if you
            use data that displays data with a data range greater than 5 decades.
    """
    # check inputs
    inputs = [negative, width, positive, max_value]
    len_param = 0.0
    for N in inputs:
        if hasattr(N, "__len__") and (not isinstance(N, str)):
            len_param += len(N) / 4
        else:  # integer values do not have len attribute
            len_param += 0.25
    # set copy of adata if inplace=False
    adata = adata if inplace else adata.copy()
    # transform every variable the same:
    if len_param == 1:
        x, y = _generate_biex_lut(
            neg=negative, width_basis=width, pos=positive, max_value=max_value
        )

        # lut_func to apply for transformation
        lut_func = interpolate.interp1d(
            x, y, kind="linear", bounds_error=False, fill_value=(np.min(y), np.max(y))
        )

        # transform adata values using the biexponential function
        adata.X = lut_func(adata.X)

    elif len_param == adata.n_vars:
        for idx, marker in enumerate(adata.var_names):
            # get correct row
            row_idx = negative.index == marker

            negative_tmp = negative[row_idx][0]
            width_tmp = width[row_idx][0]
            positive_tmp = positive[row_idx][0]
            max_value_tmp = max_value[row_idx][0]

            x, y = _generate_biex_lut(
                neg=negative_tmp,
                width_basis=width_tmp,
                pos=positive_tmp,
                max_value=max_value_tmp,
            )

            # lut_func to apply for transformation
            lut_func = interpolate.interp1d(
                x,
                y,
                kind="linear",
                bounds_error=False,
                fill_value=(np.min(y), np.max(y)),
            )

            # transform adata values using the biexponential function
            adata.X[:, idx] = lut_func(adata.X[:, idx])
    else:
        print(
            "One of the parameters has the incorrect length.               Return adata"
            " without normalising."
        )

    return None if inplace else adata


def _generate_biex_lut(
    channel_range=4096, pos=4.418540, neg=0.0, width_basis=-10, max_value=262144.000029
):
    """Creates a FlowJo compatible biex lookup table.

    Code adopted from FlowKit Python package.
    Creates a FlowJo compatible biex lookup table.

    Implementation ported from the R library cytolib, which claims to be
    directly ported from the legacy Java code from TreeStar.

    :param channel_range: Maximum positive value of the output range
    :param pos: Number of decades
    :param neg: Number of extra negative decades
    :param width_basis: Controls the amount of input range compressed in
        the zero / linear region. A higher
        width basis value will include more input values in
        the zero / linear region.
    :param max_value: maximum input value to scale
    :return: 2-column NumPy array of the LUT (column order: input, output)
    """
    ln10 = np.log(10.0)
    decades = pos
    low_scale = width_basis
    width = np.log10(-low_scale)

    decades = decades - (width / 2)

    extra = neg

    if extra < 0:
        extra = 0

    extra = extra + (width / 2)

    zero_point = int((extra * channel_range) / (extra + decades))
    zero_point = int(np.min([zero_point, channel_range / 2]))

    if zero_point > 0:
        decades = extra * channel_range / zero_point

    width = width / (2 * decades)

    maximum = max_value
    positive_range = ln10 * decades
    minimum = maximum / np.exp(positive_range)

    negative_range = _log_root(positive_range, width)

    max_channel_value = channel_range + 1
    n_points = max_channel_value

    step = (max_channel_value - 1) / (n_points - 1)

    values = np.arange(n_points)
    positive = np.exp(values / float(n_points) * positive_range)
    negative = np.exp(values / float(n_points) * -negative_range)

    # apply step to values
    values = values * step

    s = np.exp((positive_range + negative_range) * (width + extra / decades))

    negative *= s
    s = positive[zero_point] - negative[zero_point]

    positive[zero_point:n_points] = (
        positive[zero_point:n_points] - negative[zero_point:n_points]
    )
    positive[zero_point:n_points] = minimum * (positive[zero_point:n_points] - s)

    neg_range = np.arange(zero_point)
    m = 2 * zero_point - neg_range

    positive[neg_range] = -positive[m]

    return positive, values


def _log_root(b: float, w: float) -> float:
    """Helper function.

    Args:
        b (float): Upper bound
        w (float): Step parameter

    Returns:
        float: Solution to interpolation
    """
    # Code adopted from FlowKit Python package
    x_lo = 0.0
    x_hi = b
    d = (x_lo + x_hi) / 2
    dx = abs(int(x_lo - x_hi))  # type: float
    dx_last = dx
    fb = -2 * np.log(b) + w * b
    f = 2.0 * np.log(d) + w * b + fb
    df = 2 / d + w

    if w == 0:
        return b

    for i in range(100):
        if (((d - x_hi) * df - f) - ((d - x_lo) * df - f)) > 0 or abs(2 * f) > abs(
            dx_last * df
        ):
            dx = (x_hi - x_lo) / 2
            d = x_lo + dx
            if d == x_lo:
                return d
        else:
            dx = f / df
            t = d
            d -= dx
            # if dx is smaller than some precision threshold
            if d == t:
                return d

        # if dx is smaller than some precision threshold
        if abs(dx) < 1.0e-12:
            return d

        dx_last = dx
        f = 2 * np.log(d) + w * d + fb
        df = 2 / d + w
        if f < 0:
            x_lo = d
        else:
            x_hi = d

    return d
