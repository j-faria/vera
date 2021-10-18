import os
import requests
from copy import copy
import warnings
import numpy as np
import matplotlib
# matplotlib.use('pgf')
# pgf_custom_preamble = {
#     "text.usetex": True,
#     "text.latex.preamble": [r"\usepackage{hyperref}"],
#     # "pgf.preamble": [r"\usepackage{hyperref}"]
# }
# matplotlib.rcParams.update(pgf_custom_preamble)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import pickle

import lightkurve

here = os.path.dirname(__file__)

# # TOI info file
# filename = 'hlsp_tess-data-alerts_tess_phot_alert-summary-s01+s02+s03+s04_tess_v9_spoc.csv'
# f1 = os.path.join(here, 'data', filename)

# TOI info file
# filename = 'csv-file-toi-catalog.csv'
# f = os.path.join(here, 'data', filename)

# download TOI list from exoFOP
TOIurl = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=pipe"
filename = 'csv-file-toi-catalog-exofop.csv'
f = os.path.join(here, 'data', filename)


def getTOIs(update=False):
    if update or not os.path.exists(f):
        print('Downloading list of TOIs from exoFOP...')
        TOI = requests.get(TOIurl)
        open(f, 'w').write(TOI.content.decode())

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings("ignore")
        d = np.genfromtxt(f, delimiter='|', names=True, dtype=None,
                          encoding='utf8', skip_header=0, invalid_raise=False)
    TOIs = d['TOI'].astype(int)
    return list(TOIs), d


def getTOI_info(toi):
    # d = np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding='utf8')
    # TOIs = d['toi_id'].astype(int)
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings("ignore")
        d = np.genfromtxt(f, delimiter='|', names=True, dtype=None,
                          encoding='utf8', skip_header=0, invalid_raise=False)

    TOIs = d['TOI'].astype(int)
    ind = np.where(TOIs == toi)[0]
    info = d[ind]
    # r = {toi: {}}
    # r[toi]['TIC'] = d[ind]['TIC_ID'][0]
    # r[toi]['P'] = d[ind]['Period_days']
    return info


def join(lcs):
    LC = lcs[0]
    print(LC)
    for lc in lcs[1:]:
        print(lc)
        LC = LC.append(lc)
    return LC


def search_lightcurve(TIC):
    try:
        cached = pickle.load(open('CachedSearches.pickle', 'rb'))
    except FileNotFoundError:
        cached = {}

    TIC = f'TIC{TIC}'

    if TIC in cached:
        return cached[TIC]
    else:
        original = lightkurve.search_lightcurve(TIC, mission='TESS')
        # keep the cached dict manageable
        n = len(cached)
        if n > 100:
            # remove a random cached target
            cached.pop(np.random.choice(list(cached.keys())))
        cached.update({TIC: original})
        pickle.dump(cached, open('CachedSearches.pickle', 'wb'))
        return original


def add_docs(tool):
    def desc(func):
        func.__doc__ = "Showing help for %s()" % tool
        return func
    return desc


class TESS():

    def __init__(self, TIC=None, TOI=None, keep_fast=False):
        if TIC is None and TOI is None:
            raise ValueError('Provide either TIC ID or TOI number.')

        if TIC is None:  # find TOI
            if not (isinstance(TOI, int) or isinstance(TOI, str)):
                raise ValueError('TOI number should be integer or string.')

            TOI = int(TOI)

            if TOI not in getTOIs()[0]:
                raise ValueError('Cannot find TOI %d in TOI list.')

            self.info = info = getTOI_info(TOI)
            self.info_names = info.dtype.fields
        
            self.TOI = TOI
            self.TIC = info['TIC_ID'][0]

            self.period = info['Period_days']

        else:
            self.TIC = TIC
            TOIdata = getTOIs()[1]
            # print(TIC in TOIdata['TIC_ID'])
            self.period = None

        self.search = search_lightcurve(self.TIC)
        if len(self.search) == 0:
            raise ValueError(f'search_lightcurve failed for {self.TIC}')

        if not keep_fast:
            # Indices where the lightcurve name does not contain 'fast' : 20s readout
            ind = np.where(np.char.find(np.array(self.search.table['productFilename']), 'fast') < 0)[0]
            self.search = self.search[[ind]]

        self.lcs = self.search.download_all()
        self.lc = self.lcs.stitch()
        self.lc.remove_nans()
        # self.lc = join(self.lcs)
        # self.lcs = [f.flux for f in self.lcs]
        # self.lcs = [lc.remove_nans() for lc in self.lcs]

        self._undo_lcs = []
        self._undo_lc = []

    # def __getattr__(self, attr):
    #     if hasattr(self.lc, attr) and callable(getattr(self.lc, attr)):
    #         self.lc = getattr(self.lc, attr)

    def __repr__(self):
        return f'TESS(TIC {self.TIC}; {len(self.lcs)} sectors)'

    @property
    def time(self):
        return self.lc.time

    @property
    def flux(self):
        return self.lc.flux

    @property
    def flux_err(self):
        return self.lc.flux_err

    def plot(self, **kwargs):
        self.lc.plot(**kwargs)

    def errorbar(self, **kwargs):
        self.lc.errorbar(**kwargs)

    def normalize(self):
        self._undo_lcs.append(copy(self.lcs))
        self._undo_lc.append(copy(self.lc))
        self.lc = join([lc.normalize() for lc in self.lcs])

    def bin(self, *args, **kwargs):
        self._undo_lcs.append(copy(self.lcs))
        self._undo_lc.append(copy(self.lc))
        self.lc = self.lc.bin(*args, **kwargs)

    def remove_outliers(self, *args, **kwargs):
        self._undo_lcs.append(copy(self.lcs))
        self._undo_lc.append(copy(self.lc))
        self.lcs = [lc.remove_outliers(*args, **kwargs) for lc in self.lcs]
        self.lc = self.lc.remove_outliers(*args, **kwargs)

    def remove_nans(self, *args, **kwargs):
        self._undo_lcs.append(copy(self.lcs))
        self._undo_lc.append(copy(self.lc))
        self.lcs = [lc.remove_nans(*args, **kwargs) for lc in self.lcs]
        self.lc = self.lc.remove_nans(*args, **kwargs)

    def undo(self):
        if len(self._undo_lc) == 0:
            return

        self.lcs = copy(self._undo_lcs[-1])
        self._undo_lcs.pop(-1)
        self.lc = copy(self._undo_lc[-1])
        self._undo_lc.pop(-1)

    def gls(self, **kwargs):
        per = self.lc.to_periodogram(method='ls', **kwargs)
        ax = per.plot(view='period')
        ax.set_xscale('log')

        if self.period is not None:
            ymin, ymax = ax.get_ylim()
            ax.vlines(self.period, 0.95*ymax, ymax, color='m')

    def bls(self, **kwargs):
        ax = kwargs.pop('ax', None)
        per = self.lc.to_periodogram(method='bls', **kwargs)

        ax = per.plot(view='period', ax=ax)
        per_max_power = per.period_at_max_power.value
        print('Best fit period: {:.3f}'.format(per_max_power))
        # ax.set_xscale('log')

        if self.period is not None:
            ymin, ymax = ax.get_ylim()
            for period in self.period:
                ax.vlines(period, 0.95*ymax, ymax, color='m')
            ax.vlines(per_max_power, 0.95*ymax, ymax, color='b')

    def flatten(self, *args, **kwargs):
        self._undo_lcs.append(copy(self.lcs))
        self._undo_lc.append(copy(self.lc))
        self.lc = self.lc.flatten(*args, **kwargs)

    def report(self, clean_data=True):
        if clean_data:
            self.normalize()
            self.remove_outliers(sigma=15)

        size = 8.27, 11.69
        fig = plt.figure(figsize=size, constrained_layout=True)
        # fig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.1)
        # fig.suptitle(self.star)
        gs = gridspec.GridSpec(5, 3, figure=fig,
                               height_ratios=[2, 2, 1, 2, 0.1])

        ax1 = plt.subplot(gs[0, :])

        if self.TOI:
            title = 'TOI ' + str(self.TOI)
        else:
            title = 'TIC ' + str(self.TIC)

        ax1.set_title(title, loc='left')

        self.plot(ax=ax1)
        offset = 0.001
        for period, t0 in zip(self.period, self.info['Epoch_BJD']):
            if period == 0.0:
                n = 20
            else:
                n = int(self.time.ptp() / period) + 1
            transits = (t0-2457000.0) + period * np.arange(0, n)
            ax1.plot(transits, np.full_like(transits,
                                            self.flux.min() - offset), '*')
            offset += offset
            
        ax1.get_legend().remove()

        ax2 = plt.subplot(gs[1, :-1])

        if self.flux.shape[0] > 50000:
            freq = round((self.flux.shape[0] / 500.), -1)
            self.bls(frequency_factor=freq,
                     period=np.arange(0.3, 120, 0.1),
                     ax=ax2)
        else:
            self.bls(period=np.arange(0.3, max(np.ceil(self.time.ptp()), 120), 0.01),
                     ax=ax2)
        ax2.get_legend().remove()

        # phase = time / period
        for i, (period, t0, duration) in enumerate(zip(self.period,
                                                       self.info['Epoch_BJD'],
                                                       self.info['Duration_hours'])):
            print('Period | Time of transit | Duration\n', period, t0, duration)
            if period != 0.0:
                ax = plt.subplot(gs[1+i, -1])
                
                folded_lc = self.lc.fold(period=period,
                                         t0=t0-2457000)
                folded_lc.plot(ax=ax)
                dur_days = duration / 24.0
                dur_phase = dur_days / period
                
                ax.set_xlim(-3.0 * dur_phase, 3.0 * dur_phase)
                ax.get_legend().remove()


# set the docs!
def strip_doc_of_returns(doc):
    doc = doc[:doc.find('Returns')]
    return doc


TESS.remove_outliers.__doc__ = strip_doc_of_returns(
    lightkurve.TessLightCurve.remove_outliers.__doc__)
TESS.remove_nans.__doc__ = strip_doc_of_returns(
    lightkurve.TessLightCurve.remove_nans.__doc__)
