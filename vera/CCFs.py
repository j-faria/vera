import os

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import iCCF
try:
    from iCCF.chromatic import chromaticRV
except ImportError:
    chromaticRV = None


class HeaderCCF():
    def __init__(self, indicators):
        if isinstance(indicators, iCCF.Indicators):
            indicators = [indicators, ]

        self._headers = [i.HDU[0].header for i in indicators]
        self.n = len(indicators)

    def __repr__(self):
        return f'CCF headers of {self.n} files'

    def __getitem__(self, i):
        return self._headers[i]

    @property
    def keywords(self):
        """ List of all keywords available in the headers """
        return list(self._headers[0].keys())

    def find_keyword(self, kwd, raise_not_found=False, verbose=True):
        """
        Find keyword `kwd` in the headers. Can have wildcards, e.g., *CCF*
        If raise_not_found, raises KeyError if the keyword is not found.
        """
        if fits.header.Header._haswildcard(None, kwd):
            keys = []
            for H in self._headers:
                keys += list(H[kwd].keys())
            keys = list(set(keys))
            if verbose:
                print('\n'.join(keys))
            return keys
        else:
            if kwd in self._headers[0]:
                return [kwd]
            else:
                if raise_not_found:
                    raise KeyError(f'{kwd} not found')
                if verbose:
                    print(f'Keyword {kwd} not found')

    def get(self, *kwds, add_bjd=False):
        """
        Get numpy array with the values of the keywords `kwds` for all headers.
        If add_bjd, the first column of the array will be the 'MJD-OBS' value.
        Examples:
            get('ESO QC CCF RV', 'ESO QC CCF RV ERROR')
            get('ESO QC CCF RV', add_bjd=True)
        """
        if add_bjd:
            kwds = list(kwds)
            kwds.insert(0, 'MJD-OBS')
            kwds = tuple(kwds)

        if len(kwds) == 1:
            kwd = kwds[0]
            # dtype = type(self._headers[0][kwd])
            vals = []
            for h in self._headers:
                try:
                    if '*' in kwd:
                        vals.append(h[kwd][0])
                    else:
                        vals.append(h[kwd])
                except KeyError:
                    vals.append(None)

            return np.array(vals)

        else:
            r = []
            dtypes = []

            for kwd in kwds:
                try:
                    dtype = type(self._headers[0][kwd])
                    decode = False

                    if dtype is str:
                        dtype = 'S%d' % len(self._headers[0][kwd])
                        decode = True

                    kwargs = dict(dtype=dtype, count=self.n)
                    arr = np.fromiter((h[kwd] for h in self._headers), **kwargs)

                    if decode:
                        arr = np.char.decode(arr)

                    r.append(arr)
                    dtypes.append(dtype)

                except Exception as e:
                    print(e)
                    return

            dtype = ', '.join([v.dtype.str for v in r])
            rarray = np.zeros(self.n, dtype=dtype)
            rarray.dtype.names = kwds

            for i, name in enumerate(rarray.dtype.names):
                rarray[name] = r[i]

            return rarray

    def tofile(self, filename, *kwds, delimiter=' ', comments='#'):
        """
        Save the values of the keywords `kwds` from each header to `filename`.

        Parameters
        ----------
        filename : str
            The name of the file to save. 
        kwds : sequence
            Keywords to extract from the headers and save to the file.
        delimiter : str, optional
            String or character separating columns.
        comments : str, optional
            String that will be prepended to the header in the first line.
        """
        if os.path.exists(filename):
            print(f'File {filename} exists. Replace? [y/N] ', end='')
            answer = input()
            if answer != 'y':
                return

        r = self.get(*kwds)

        with open(filename, 'w') as f:
            f.write(comments + ' ' + delimiter.join(kwds) + '\n')
            for rr in r:
                f.write(delimiter.join(map(str, rr)) + '\n')


def chromatic_plot_main(rv):
    fig, axs = plt.subplots(3 + 1, 2, constrained_layout=True)

    axs = axs.ravel()

    indices_plots = np.arange(0, 8, 2)
    indices_periodograms = np.arange(1, 9, 2)

    kw = dict(right_ticks=False, legend=False)
    ekw = dict(fmt='o', ms=2)

    # axs[indices_plots[0]].errorbar(rv.time, rv.vrad - rv.vrad.mean(), rv.svrad,
    #                                color='k', **kw)

    rv.plot(ax=axs[indices_plots[0]], color='k', **kw, **ekw)

    rv.blue.plot(ax=axs[indices_plots[1]], color='b', **kw, **ekw)
    rv.mid.plot(ax=axs[indices_plots[2]], color='g', **kw, **ekw)
    rv.red.plot(ax=axs[indices_plots[3]], color='r', **kw, **ekw)

    # for ax in axs[indices_plots]:
    #     ax.get_legend().remove()

    # axs[indices_plots[1]].errorbar(rv.time, rv.blue.vrad,
    #                                rv.blue.svrad, color='b', **kw)
    # axs[indices_plots[2]].errorbar(rv.time, rv.midRV - rv.midRV.mean(),
    #                                rv.mid.svrad, color='g', **kw)
    # axs[indices_plots[3]].errorbar(rv.time, rv.redRV - rv.redRV.mean(),
    #                                rv.red.svrad, color='r', **kw)

    # from astropy.timeseries import LombScargle

    kw = dict(frequency=False, HZ=True, bootstrap=False, recompute=True,
              legend=False, plot_data_with_offsets=False)

    systems = (rv, rv.blue, rv.mid, rv.red)

    for s, axd, axp in zip(systems, axs[indices_plots],
                           axs[indices_periodograms]):
        s.gls(ax=axp, **kw)
        if s.GLS['gatspy']:
            m = s.GLS['model']
            for filt, ym in zip(np.unique(m.filts), m.ymean_by_filt_):
                mask = m.filts == filt
                axd.hlines(ym, m.t[mask].min(), m.t[mask].max(), ls='--')

    for ax in axs[indices_plots][:-1]:
        ax.set_xlabel('')
    for ax in axs[indices_periodograms][:-1]:
        ax.set_xlabel('')

    return fig, axs
