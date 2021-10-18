import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle

from .binit import binRV
from .query_ASASSN import get_lightcurve as get_lightcurve_ASASSN
from .query_superWASP import get_lightcurve as get_lightcurve_superWASP


class photometry:
    def __init__(self, target, time, flux, flux_err, verbose=True):
        self.target = target
        self.time, self.flux, self.flux_err = time, flux, flux_err
        self.verbose = verbose
        self.source = 'photometry'

        self.c_time = self.time.copy()
        self.c_flux = self.flux.copy()
        self.c_flux_err = self.flux_err.copy()
        self.mask = np.ones_like(self.flux, dtype=bool)

    @classmethod
    def superWASP(cls, filename, verbose=True):
        # read the lightcurve
        data = np.genfromtxt(filename, delimiter=',', names=True)
        target = filename[:-4]
        N = data.size
        time = data['HJD']
        time -= 24e5
        mag = data['magnitude']
        median_mag = np.median(mag)
        flux = np.negative(mag - median_mag) + median_mag
        flux_err = data['magnitude_error']
        c = cls(target, time, flux, flux_err)
        c.source = 'superWASP'
        return c

    @classmethod
    def ASAS_SN(cls, filename, verbose=True):
        # read the lightcurve
        data = np.genfromtxt(filename, delimiter=',', names=True)
        if len(data) == 0:
            raise ValueError(f'No data in {filename}')

        target = filename[:-4]
        time = data['hjd']
        time -= 24e5
        mag = data['mag']
        median_mag = np.median(mag)
        flux = np.negative(mag - median_mag) + median_mag
        flux_err = data['mag_err']
        c = cls(target, time, flux, flux_err)
        c.source = 'ASAS-SN'
        return c

    @property
    def N(self):
        return self.c_time.size

    def __repr__(self):
        return f'{self.source}({self.target}, {self.N} points)'

    @classmethod
    def query_object(cls, star, catalogue='superWASP', verbose=True):
        """ 
        Query photometry catalogues for a given star

        Arguments
        ---------
        star : str
            The star name
        catalogue : str, either 'superWASP' or 'ASAS-SN'
            The catalogue to query
        verbose : bool, default True
            Be verbose when querying, downloading data
        """
        if catalogue == 'superWASP':
            filename = get_lightcurve_superWASP(star, verbose)
            return photometry.superWASP(filename, verbose=verbose)
        elif catalogue == 'ASAS-SN':
            filename = get_lightcurve_ASASSN(star, verbose)
            return photometry.ASAS_SN(filename, verbose=verbose)

    def bin(self):
        bt, by, be = binRV(self.c_time, self.c_flux, self.c_flux_err)
        self.c_time, self.c_flux, self.c_flux_err = bt, by, be

    def sigmaclip(self, start_sigma=4, iterative=True, step=0.8, niter=5):
        def plotit(original, mask):
            plt.close('sigmaclip_fig')
            fig, ax = plt.subplots(1, 1, num='sigmaclip_fig',
                                   constrained_layout=True)
            ax.errorbar(self.time, original, fmt='o', ms=2, alpha=0.2)
            ax.plot(self.time[~mask], original[~mask], 'x', color='r', ms=2)
            plt.show()

        original = self.flux.copy()
        it, start = 0, start_sigma
        sigma = start
        msg = 'sigma={:.2f}  continue(c) stop(s) : '
        while it < niter:
            clipped, lo, up = stats.sigmaclip(original, low=sigma, high=sigma)
            mask = (original > lo) & (original < up)
            if iterative:
                plotit(original, mask)
                go_on = input(msg.format(sigma))
                if go_on == 's':
                    break
            sigma *= step
            it += 1

        self.c_time = self.time[mask]
        self.c_flux = clipped
        self.c_flux_err = self.flux_err[mask]
        self.mask = mask
        return clipped, lo, up

    def detrend(self, plot=True, degree=1, weigh=True):
        # if self.verbose:
        #     print('Removing trend')
        t, y, e = self.c_time, self.c_flux, self.c_flux_err

        if weigh:
            fitp = np.polyfit(t, y, degree, w=1/e)
        else:
            fitp = np.polyfit(t, y, degree)

        print(f'coefficients: {fitp}')

        if plot:
            ax, _ = self.plot()
            tt = np.linspace(t.min(), t.max(), 1000)
            # max_zorder = max([l.get_zorder() for l in ax.get_lines()])
            ax.plot(tt, np.polyval(fitp, tt), color='k', lw=3, zorder=3)

        y = y - np.polyval(fitp, t)
        self.c_flux = y

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
        else:
            fig = ax.figure
        markers, caps, bars = ax.errorbar(self.c_time, self.c_flux,
                                          self.c_flux_err, fmt='o', ms=2,
                                          alpha=0.6, ecolor='k', capsize=0)
        [bar.set_alpha(0.2) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        ax.set(ylabel='-mag', xlabel='JD [days]')
        return ax, fig

    def gls(self, ax=None, samples_per_peak=10, peaks=True, fap_limit=0.1):
        model = LombScargle(self.c_time, self.c_flux, self.c_flux_err)
        f, p = model.autopower(samples_per_peak=samples_per_peak)
        if (p < 0).any():
            f, p = model.autopower(samples_per_peak=samples_per_peak,
                                   method='cython')

        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)

        ax.semilogx(1 / f, p)
        ax.hlines(model.false_alarm_level([0.01, 0.1]), *ax.get_xlim(),
                  color='k', alpha=0.2, ls='--')

        if peaks:
            fap = model.false_alarm_level(fap_limit)
            i = find_peaks(p, height=float(fap))[0]
            if len(i) > 0 and len(i) < 5:
                ax.semilogx(1 / f[i], p[i], 'o', ms=3)
                title = f'peaks: {np.round(1/f[i], 1)}'
                ax.set_title(title, loc='right', fontsize=8)


        return model, (1 / f, p)
