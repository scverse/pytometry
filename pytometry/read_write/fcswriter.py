#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
import pathlib
import struct
import warnings
import numpy as np

# Replaced by manually defined version
# from ._version import version

'''
This is a modified version of the fcswrite package from
https://github.com/ZELLMECHANIK-DRESDEN/fcswrite
'''

'''BSD 3-Clause License

Copyright (c) 2016, ZELLMECHANIK DRESDEN
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.'''

"""Write .fcs files for flow cytometry"""

'''Replaced by manually defined version'''


# from ._version import version


def write_fcs(filename, ch_shortnames, chn_names, data,
              text_kw_pr=None,
              endianness="big",
              compat_chn_names=True,
              compat_copy=True,
              compat_negative=True,
              compat_percent=True,
              compat_max_int16=10000):
    """Write numpy data to an .fcs file (FCS3.0 file format)


    Parameters
    ----------
    filename: str or pathlib.Path
        Path to the output .fcs file
    chn_names: list of str, length C
        Names of the output channels
    ch_shortnames: list of str, length C
        Shortnames of the output channels
    data: 2d ndarray of shape (N,C)
        The numpy array data to store as .fcs file format.
    text_kw_pr: dict
        User-defined, optional key-value pairs that are stored
        in the primary TEXT segment
    endianness: str
        Set to "little" or "big" to define the byte order used.
    compat_chn_names: bool
        Compatibility mode for 3rd party flow analysis software:
        The characters " ", "?", and "_" are removed in the output
        channel names.
    compat_copy: bool
        Do not override the input array `data` when modified in
        compatibility mode.
    compat_negative: bool
        Compatibliity mode for 3rd party flow analysis software:
        Flip the sign of `data` if its mean is smaller than zero.
    compat_percent: bool
        Compatibliity mode for 3rd party flow analysis software:
        If a column in `data` contains values only between 0 and 1,
        they are multiplied by 100.
    compat_max_int16: int
        Compatibliity mode for 3rd party flow analysis software:
        If a column in `data` has a maximum above this value,
        then the display-maximum is set to 2**15.

    Notes
    -----

    - These commonly used unicode characters are replaced: "µ", "²"
    - If the input data contain NaN values, the corresponding rows
      are excluded due to incompatibility with the FCS file format.

    """
    # determine version manually
    if text_kw_pr is None:
        text_kw_pr = {}
    version = '0.1'
    filename = pathlib.Path(filename)
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=float)
    # remove rows with nan values
    nanrows = np.isnan(data).any(axis=1)
    if np.sum(nanrows):
        msg = "Rows containing NaNs are not written to {}!".format(filename)
        warnings.warn(msg)
        data = data[~nanrows]
    if endianness not in ["little", "big"]:
        raise ValueError("`endianness` must be 'little' or 'big'!")

    msg = "length of `chn_names` must match length of 2nd axis of `data`"
    assert len(chn_names) == data.shape[1], msg

    rpl = [["µ", "u"],
           ["²", "2"],
           ]

    if compat_chn_names:
        # Compatibility mode: Clean up headers.
        rpl += [[" ", ""],
                ["?", ""],
                ["_", ""],
                ]

    for ii in range(len(chn_names)):
        for (a, b) in rpl:
            chn_names[ii] = chn_names[ii].replace(a, b)

    # Data with values between 0 and 1
    pcnt_cands = []
    for ch in range(data.shape[1]):
        if data[:, ch].min() >= 0 and data[:, ch].max() <= 1:
            pcnt_cands.append(ch)
    if compat_percent and pcnt_cands:
        # Compatibility mode: Scale values b/w 0 and 1 to percent
        if compat_copy:
            # copy if requested
            data = data.copy()
        for ch in pcnt_cands:
            data[:, ch] *= 100

    if compat_negative:
        toflip = []
        for ch in range(data.shape[1]):
            if np.mean(data[:, ch]) < 0:
                toflip.append(ch)
        if len(toflip):
            if compat_copy:
                # copy if requested
                data = data.copy()
            for ch in toflip:
                data[:, ch] *= -1

    # DATA segment
    flattendata = data.flatten().tolist()
    result_data = struct.pack('>%sf' % len(flattendata), *flattendata)

    # TEXT segment
    header_size = 256

    if endianness == "little":
        # use little endian
        byteord = '1,2,3,4'
    else:
        # use big endian
        byteord = '4,3,2,1'
    meta_text = '{seperator}$BEGINANALYSIS{seperator}0{seperator}$ENDANALYSIS{seperator}0'
    meta_text += '{seperator}$BEGINSTEXT{seperator}0{seperator}$ENDSTEXT{seperator}0'
    # Add placeholders for $BEGINDATA and $ENDDATA, because we don't
    # know yet how long TEXT is.
    meta_text += '{seperator}$BEGINDATA{seperator}{data_start_byte}{seperator}$ENDDATA{seperator}{data_end_byte}'
    meta_text += '{seperator}$BYTEORD{seperator}{0}{seperator}$DATATYPE{seperator}F'.format(byteord, seperator=chr(12))
    meta_text += '{seperator}$MODE{seperator}L{seperator}$NEXTDATA{seperator}0{seperator}$TOT{seperator}' \
                 '{0}'.format(data.shape[0], seperator=chr(12))
    meta_text += '{seperator}$PAR{seperator}{0}'.format(data.shape[1], seperator=chr(12))
    # Add fcswrite version
    meta_text += '{seperator}fcswrite version{seperator}{0}'.format(version, seperator=chr(12))
    # Add additional key-value pairs by the user
    for key in sorted(text_kw_pr.keys()):
        meta_text += '{seperator}{0}{seperator}{1}'.format(key, text_kw_pr[key], seperator=chr(12))
    # Check for content of data columns and set range
    for jj in range(data.shape[1]):
        # Set data maximum to that of int16
        if (compat_max_int16
                and compat_max_int16 < np.max(data[:, jj]) < 2 ** 15):
            pnrange = int(2 ** 15)
        # Set range for data with values between 0 and 1
        elif jj in pcnt_cands:
            if compat_percent:  # scaled to 100%
                pnrange = 100
            else:  # not scaled
                pnrange = 1
        # default: set range to maxium value found in column
        else:
            pnrange = int(abs(np.max(data[:, jj])))
        # TODO:
        # - Set log/lin
        fmt_str = '{seperator}$P{0}B{seperator}32{seperator}$P{0}E{seperator}0,0{seperator}$P{0}N{seperator}{1}' \
                  '{seperator}$P{0}S{seperator}{2}{seperator}$P{0}R{seperator}{3}{seperator}$P{0}D{seperator}Linear'
        meta_text += fmt_str.format(jj + 1, ch_shortnames[jj], chn_names[jj], pnrange, seperator=chr(12))
    meta_text += '{seperator}'

    # SET $BEGINDATA and $ENDDATA using the current size of TEXT plus padding.
    text_padding = 47  # for visual separation and safety
    data_start_byte = header_size + len(meta_text) + text_padding
    data_end_byte = data_start_byte + len(result_data) - 1
    meta_text = meta_text.format(data_start_byte=data_start_byte, data_end_byte=data_end_byte, seperator=chr(12))
    lentxt = len(meta_text)
    # Pad TEXT segment with spaces until data_start_byte
    meta_text = meta_text.ljust(data_start_byte - header_size, " ")

    # HEADER segment
    ver = 'FCS3.0'

    textfirst = '{0: >8}'.format(header_size)
    textlast = '{0: >8}'.format(lentxt + header_size - 1)

    # Starting with FCS 3.0, data segment can end beyond byte 99,999,999,
    # in which case a zero is written in each of the two header fields (the
    # values are given in the text segment keywords $BEGINDATA and $ENDDATA)
    if data_end_byte <= 99999999:
        datafirst = '{0: >8}'.format(data_start_byte)
        datalast = '{0: >8}'.format(data_end_byte)
    else:
        datafirst = '{0: >8}'.format(0)
        datalast = '{0: >8}'.format(0)

    anafirst = '{0: >8}'.format(0)
    analast = '{0: >8}'.format(0)

    header = '{0: <256}'.format(ver + '    ' + textfirst
                                + textlast
                                + datafirst
                                + datalast
                                + anafirst
                                + analast)

    # Write data
    with filename.open("wb") as fd:
        fd.write(header.encode("UTF-8", "replace"))
        fd.write(meta_text.encode("UTF-8", "replace"))
        fd.write(result_data)
        fd.write(b'00000000')
