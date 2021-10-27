import os
from glob import glob
from pprint import pprint
from requests import RequestException
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

from cached_property import cached_property

from .RV import RV
from .CCFs import iCCF, chromaticRV, HeaderCCF, chromatic_plot_main
from .spectra import Spectrum, Spectrum2D
from .query_dace import get_CCFs, get_spectra, get_E2DS, translate
from .utils import (yel, yelb, red as cl_red, blue as cl_blue, info)

here = os.path.dirname(os.path.abspath(__file__))


def escape(target):
    if 'proxima' in target.lower():
        target = 'Proxima'
    if target[0].isdigit():
        target = '_' + target
    target = target.replace('-', '_').replace('+', '_').replace('.', '_')
    return target


def deescape(target):
    if target in translate:
        target = translate[target]
    else:
        target = target.replace('K2_', 'K2-')
        target = target.replace('TOI_', 'TOI-')
        target = target.replace('BD_', 'BD+')
        target = target.replace('KOBE_', 'KOBE-')
        target = target.replace('_', '')

    return target


class DACERV(RV):
    """A set of RV observations for a DACE target"""

    def __init__(self, *args, **kwargs):
        super(DACERV, self).__init__(*args, **kwargs)
        self._ccfs_enabled = False
        self._spec_enabled = False

    def __repr__(self):
        n1 = self.mask.sum()
        n2 = self.mask.size
        s = f'RV({self.star}; {n1} points'
        if n1 != n2:
            s += f' [{n2-n1}/{n2} masked]'
        if self.chromatic:
            s += ', chromatic RVs enabled'
        if self.spectroscopic:
            s += ', spectra enabled'
        s += ')'
        return s

    @property
    def info(self):
        from .utils import print_system_info
        print_system_info(self)

    def bin(self, *args, **kwargs):
        try:
            if self.chromatic and not '_child' in kwargs:
                self.cRV.bin(self.night_indices)
                del self.blue, self.mid, self.red
        except AttributeError:
            pass

        super(DACERV, self).bin(*args, **kwargs)

    def _download_spectra(self, s1d=True, s2d=True, **kwargs):
        if self.verbose:
            print('Will try to download ESPRESSO S1D and S2D from DACE')

        if s1d:
            if self.verbose:
                print('- S1D:')

            self.spec_dir = get_spectra(self.star, **kwargs)

        if s2d:
            if self.verbose:
                print('- S2D:')

            self.spec_dir = get_E2DS(self.star, **kwargs)

        if self.spec_dir == 'stopped':
            self._spec_enabled = False
            return
        self._spec_enabled = True

    def enable_spectra(self, directory=None, s1d=True, s2d=True,
                       all_instruments=True, **kwargs):
        """
        Enable spectroscopic calculations by trying to download all available
        S1D and E2DS files from DACE.

        Arguments
        ---------
        directory: str (optional)
            Directory where to store the downloaded files. By default this is
            the star name followed by "_DACE_downloads"
        s1d: bool (optional, default True)
            Download S1D files
        s2d: bool (optional, default True)
            Download S2D files
        all_instruments: bool (optional, default True)
            Wheter to try to download CCFs for all available instruments. If
            set to False, can provide an `instrument` argument to select the
            instrument.
        limit: int (optional)
            Set a maximum number of files to download (per instrument)
        """

        # maybe spectra is enabled but with fewer spectra
        if self._spec_enabled:
            if 'limit' in kwargs:
                nn = len(self.spectra) / len(self.instruments)
                if kwargs['limit'] > nn:
                    self._spec_enabled = False

        if not self._spec_enabled:
            kw = dict(directory=directory)
            kw = {**kw, **kwargs}
            if all_instruments:
                kw['instrument'] = self.instruments

            self._download_spectra(s1d=s1d, s2d=s2d, **kw)
            if not self._spec_enabled:  # download_spectra failed/stopped
                print(cl_red | 'ERROR: spectra not enabled')
                return

        if s1d:
            s1d_files = glob(os.path.join(self.spec_dir, '*_S1D_A.fits'))
            if len(s1d_files) == 0:
                s1d_files = glob(os.path.join(self.spec_dir, '*_s1d_A.fits'))

            if len(s1d_files) < self.time.size:
                c = f'({len(s1d_files)} / {self.time.size})'
                print(yelb | 'Warning:', end=' ')
                print(yel | f'there are less S1D files than observations {c}')

            if len(s1d_files) == 0:
                return

            load = True
            try:
                load = len(self.spectra) != len(s1d_files)
            except AttributeError:
                pass
            if load:
                print(cl_blue | 'Loading S1D spectra...', flush=True)
                self.spectra = Spectrum.from_file(s1d_files)
                print(cl_blue | 'median-dividing...', flush=True)
                for s in self.spectra:
                    s.median_divide()
            return

        if s2d:
            s2d_files = glob(os.path.join(self.spec_dir, '*_S2D_A.fits'))
            if len(s2d_files) == 0:
                s2d_files = glob(os.path.join(self.spec_dir, '*_e2ds_A.fits'))
            if len(s2d_files) < self.time.size:
                c = f'({len(s2d_files)} / {self.time.size})'
                print(yelb | 'Warning:', end=' ')
                print(yel | f'there are less S2D files than observations {c}')

            load = True
            try:
                load = len(self.spectra) != len(s2d_files)
            except AttributeError:
                pass
            if load:
                info('Loading S2D spectra...')
                self.spectra = Spectrum2D.from_file(s2d_files)

        # self.cRV = chromaticRV(indicators)
        # self._round_chromatic_time()
        # self.headers = HeaderCCF(self.cRV.I)

    def _round_chromatic_time(self):
        sig_digits = len(str(self.time[0]).split('.')[1])
        self.cRV.time = self.cRV.time.round(sig_digits)

    def _download_CCFs(self, **kwargs):
        if self.verbose:
            info('Will try to download CCFs from DACE')

        self.ccf_dir = get_CCFs(self.star, **kwargs)

        if self.ccf_dir in (None, 'stopped'):
            self._ccfs_enabled = False
            return
        self._ccfs_enabled = True

    def enable_chromatic(self, directory=None, all_instruments=True, **kwargs):
        """
        Enable chromatic RV calculations by trying to download CCF files.

        Arguments
        ---------
        directory: str (optional)
            Directory where to store the downloaded files. By default this is
            the star name followed by "_DACE_downloads"
        all_instruments: bool (optional, default True)
            Wheter to try to download CCFs for all available instruments
        """
        if chromaticRV is None:
            raise NotImplementedError

        if not self._ccfs_enabled:
            kw = dict(directory=directory)
            if all_instruments:
                kw['instrument'] = self.instruments
            kw = {**kw, **kwargs}

            self._download_CCFs(**kw)

            if not self._ccfs_enabled:  # download_CCFs failed or stopped
                print(cl_red | 'ERROR: chromatic not enabled')
                return

        ccf_files = glob(os.path.join(self.ccf_dir, '*CCF_SKYSUB*.fits'))
        # ccf_files = []
        if len(ccf_files) == 0:
            ccf_files = []
            for end in ('*CCF*', '*ccf*A.fits'):
                ccf_files.extend(glob(os.path.join(self.ccf_dir, end)))

        if len(ccf_files) < self.time.size:
            counts = f'({len(ccf_files)} / {self.time.size})'
            print(yelb | 'Warning:', end=' ')
            print(yel | f'there are less CCF files than observations {counts}')

        if hasattr(self, 'cRV') and len(ccf_files) == self.cRV.n:
            if self.verbose:
                info('chromatic enabled!')
            return

        info('Loading CCFs...')
        indicators = iCCF.Indicators.from_file(
            ccf_files, guess_instrument=True, sort_bjd=True)
        self.cRV = chromaticRV(indicators)
        self._round_chromatic_time()
        self.headers = HeaderCCF(self.cRV.I)
        if self.verbose:
            info('chromatic enabled!')

    @property
    def chromatic(self):
        return self._ccfs_enabled

    @property
    def spectroscopic(self):
        return self._spec_enabled

    def _build_chromatic_RV(self, which):
        # if self is binned nightly, there might be more CCFs than time.size
        if self.is_binned:
            time = self._unbinned['time']
            svrad = self._unbinned['svrad']
            mask = self._unbinned['mask']
            obs = self._unbinned['obs']
        else:
            time = self.time
            svrad = self.svrad
            mask = self.mask
            obs = self.obs

        # this finds the mask for times that have a CCF
        # (because sometimes there are less CCFs)
        inboth = np.isin(time, self.cRV.time.round(6))
        ones = np.ones_like(time[inboth])

        #! since chromatic RVs are in km/s (measured directly from the CCF)
        #! make sure the errors we supply here are also in km/s
        if self.did_ms:
            svrad = svrad * 1e-3

        vrad = {
            'blue': self.cRV.blueRV,
            'mid': self.cRV.midRV,
            'red': self.cRV.redRV,
        }
        vrad = vrad[which]

        svrad = {
            'blue': self.cRV._blueRVerror,
            'mid': self.cRV._midRVerror,
            'red': self.cRV._redRVerror
        }
        svrad = svrad[which]
        if svrad is None:
            svrad = svrad[inboth]

        fwhm = {
            'blue': self.cRV._blueFWHM,
            'mid': self.cRV._midFWHM,
            'red': self.cRV._redFWHM,
        }
        fwhm = fwhm[which]

        efwhm = {
            'blue': self.cRV._blueFWHMerror,
            'mid': self.cRV._midFWHMerror,
            'red': self.cRV._redFWHMerror
        }
        efwhm = efwhm[which]
        if efwhm is None:
            efwhm = efwhm[inboth]


        rv = RV.from_arrays(time[inboth], vrad, svrad, fwhm, efwhm,
                            verbose=self.verbose, star=self.star,
                            sigmaclip=False, adjust_means=self.did_adjustmeans,
                            ms=self.did_ms, tess=False)

        rv.mask = mask[inboth]
        rv.obs = obs[inboth]
        rv.instruments = self.instruments
        rv.pipelines = self.pipelines

        # now bin the chromatic RVs if self is also binned
        if self.is_binned:
            rv.bin(_child=True)

        return rv

    @cached_property
    def blue(self):
        assert self.chromatic, \
            'chromatic RVs not enabled, run .enable_chromatic()'
        return self._build_chromatic_RV('blue')

    @cached_property
    def mid(self):
        assert self.chromatic, \
            'chromatic RVs not enabled, run .enable_chromatic()'
        return self._build_chromatic_RV('mid')

    @cached_property
    def red(self):
        assert self.chromatic, \
            'chromatic RVs not enabled, run .enable_chromatic()'
        return self._build_chromatic_RV('red')

    def save_chromatic_rdb(self, filename):
        header = 'jdb\tvrad\tsvrad\tvblue\tsvblue\tvmid\tsvmid\tvred\tsvred\tfwhm\tsfwhm'
        header += '\n' + '\t'.join(['-'*len(s) for s in header.split('\t')])
        fmt = ['%7.5f'] + 10*['%5.3f']

        data = np.c_[self.time, self.vrad, self.svrad,
                     self.blue.vrad, self.blue.svrad,
                     self.mid.vrad, self.mid.svrad,
                     self.red.vrad, self.red.svrad,
                     self.fwhm, self.efwhm]

        np.savetxt(filename, data, fmt=fmt, comments='', header=header,
                   delimiter='\t')

    def chromatic_plot(self):
        assert self.chromatic, \
            'chromatic RVs not enabled, run .enable_chromatic()'

        fig, axs = chromatic_plot_main(self)
        fig.set_figwidth(9)
        fig.set_figheight(10)

        ploti = np.arange(0, 8, 2)  # indices of RV plots
        peri = np.arange(1, 8, 2)  # indices of periodogram plots

        axs[0].get_shared_x_axes().join(*axs[ploti])

        axs[ploti[1]].get_shared_y_axes().join(*axs[ploti[1:]])

        axs[1].get_shared_x_axes().join(*axs[peri])

        if self.tess is not None and self.tess.period is not None:
            for ax in axs[peri]:
                y1, y2 = ax.get_ylim()
                h = 0.1 * abs(y2 - y1)
                ax.vlines(self.tess.period, ymin=y2 - h, ymax=y2, color='m',
                          alpha=0.6, label='planets')

        axs[0].set_title(r'full $\lambda$ range', loc='left', fontsize=8)

        names = ('blue', 'mid', 'red')
        for i, (ax, name) in enumerate(zip(axs[ploti[1:]], names)):
            ax.set_title(name + rf' $\lambda$={self.cRV.bands[i]} nm',
                         loc='left', fontsize=8)

        for ax in axs[ploti]:
            ax.set_ylabel('RV [m/s]')
        for ax in axs[peri]:
            ax.set_ylabel('Power')

        def roundup(a, digits=0):
            n = 10**-digits
            return round(ceil(a / n) * n, digits)

        # maxy = roundup(max([ax.get_ylim()[1] for ax in axs[ploti]]), 1)
        # miny = roundup(min([ax.get_ylim()[0] for ax in axs[ploti]]), 1)
        # for ax in axs[ploti]:
        #     ax.set_ylim(miny, maxy)

        maxy = roundup(max([ax.get_ylim()[1] for ax in axs[peri]]), 1)
        for ax in axs[peri]:
            ax.set_ylim(0, maxy)

        #     if self.prot and not np.isnan(self.prot):
        #         # legend = legend & True
        #         y1, y2 = ax.get_ylim()
        #         h = 0.05 * abs(y2 - y1)
        #         ax.vlines(self.prot, ymin=y2-2*h, ymax=y2-h, color='r',
        #                   alpha=0.6, lw=2)
        #         ax.plot(self.prot, y2-h, 'x', color='r', label=r'P$_{\rm rot}$')

        # fig.tight_layout()

        plt.show()
        return fig, axs

    def kima(self, directory=None, GUI=True, ESPRESSO_fiber_offset=True):
        # from pykima import kimaGUI, make_template

        star = ''.join(self.star.split())
        d = star + '_kima_analysis' if directory is None else directory
        self._kima_directory = d

        if not os.path.exists(d):
            print(yelb | 'Created', d)
            os.mkdir(d)

        # save data to directory
        self.save(prefix=d, save_indicators=True)

        if GUI:
            cmd = f'kima-gui {d}'
            os.system(cmd)
        else:
            create_dir = not os.path.exists(os.path.join(d, 'kima_setup.cpp'))
            self._kima(directory=d, create_dir=create_dir, edit=False)

    def save_figures(self):
        folder = f'{self.star}_figures'
        if not os.path.exists(folder):
            os.mkdir(folder)

        # plot_both
        fig, axs = self.plot_both(show_today=True, right_ticks=False)
        print(os.path.join(folder, 'plot_both.png'))
        fig.savefig(os.path.join(folder, 'plot_both.png'), dpi=200)
        # gls_both
        fig, axs = self.gls_both(HZ=True)
        axs[1].set_title('FWHM', loc='left')
        print(os.path.join(folder, 'gls_both.png'))
        fig.savefig(os.path.join(folder, 'gls_both.png'), dpi=200)

        plt.close('all')


class DACE():
    """
    This class holds information about DACE targets.

    To access the RVs of a given target T use `DACE.T`
    All symbols not allowed by Python (-, +, or .) have been replaced with an
    underscore _. For target names starting with a number, start with an _.
    Example: 'TOI-123' -> DACE.TOI_123, '51Peg' -> DACE._51Peg

    .target : RV
        Instance of the `RV` class for a given target
    """

    # kwargs for RV with their default values
    ESPRESSO_only = False  # only load ESPRESSO RVs
    local_first = False  # try to read local files before querying DACE
    verbose = True  # what it says
    bin_nightly = True  # bin the observations nightly
    sigmaclip = True  # sigma-clip RVs, FWHM, and other observations
    maxerror = 10  # max. RV error allows, mask points with larger errors
    adjust_means = True
    keep_pipeline_versions = False  # keep both ESPRESSO pipeline versions
    remove_ESPRESSO_commissioning = True  # remove RVs before commissioning
    download_TESS = False  # try to download TESS data
    ESPRESSO_fiber_offset = False
    # remove_secular_acceleration = True  # subtract secular acceleration from RVs
    remove_secular_acceleration = ('HARPS', )

    def __init__(self):
        self._print_errors = True
        self._attributes = set()
        self._ignore_attrs = list(self.__dict__.keys())

    @property
    def _EO(self):
        return self.ESPRESSO_only

    @property
    def settings(self):
        msg = "Use .set(setting=True/False) to change each setting.\n"
        msg += "For example DACE.set(bin_nightly=False, verbose=True)\n"
        print(msg)
        pprint(self._kwargs)

    @property
    def _kwargs(self):
        k = {
            'verbose': self.verbose,
            'ESPRESSO_only': self.ESPRESSO_only,
            'bin': self.bin_nightly,
            'sigmaclip': self.sigmaclip,
            'maxerror': self.maxerror,
            'adjust_means': self.adjust_means,
            'keep_pipeline_versions': self.keep_pipeline_versions,
            'remove_secular_acceleration': self.remove_secular_acceleration,
            'tess': self.download_TESS,
        }
        return k

    def set(self, verbose=None, ESPRESSO_only=None, bin_nightly=None,
            adjust_means=None, local_first=None, sigmaclip=None, maxerror=None,
            download_TESS=None, keep_pipeline_versions=None,
            remove_secular_acceleration=None,
            reload_all=True):

        def _not_none_and_different(val, name):
            return val is not None and val != getattr(self, name)

        change = False

        if _not_none_and_different(verbose, 'verbose'):
            self.verbose = verbose
            change = True

        if _not_none_and_different(ESPRESSO_only, 'ESPRESSO_only'):
            self.ESPRESSO_only = ESPRESSO_only
            change = True

        if _not_none_and_different(bin_nightly, 'bin_nightly'):
            self.bin_nightly = bin_nightly
            change = True

        if _not_none_and_different(adjust_means, 'adjust_means'):
            self.adjust_means = adjust_means
            change = True

        if _not_none_and_different(local_first, 'local_first'):
            self.local_first = local_first
            change = True

        if _not_none_and_different(sigmaclip, 'sigmaclip'):
            self.sigmaclip = sigmaclip
            change = True

        if _not_none_and_different(maxerror, 'maxerror'):
            self.maxerror = maxerror
            change = True

        if _not_none_and_different(keep_pipeline_versions,
                                   'keep_pipeline_versions'):
            self.keep_pipeline_versions = keep_pipeline_versions
            change = True

        if _not_none_and_different(download_TESS, 'download_TESS'):
            self.download_TESS = download_TESS
            change = True

        if _not_none_and_different(remove_secular_acceleration,
                                   'remove_secular_acceleration'):
            self.remove_secular_acceleration = remove_secular_acceleration
            change = True

        if change and reload_all:
            self._print_errors = False
            self.reload()
            self._print_errors = True


    def reload(self, star=None):
        if star is None:
            stars = self._attributes
        else:
            stars = [star]

        for star in stars:
            escaped_star = escape(star)
            if self.verbose:
                if escaped_star != star:
                    print(f'reloading {star} ({escaped_star})')
                else:
                    print(f'reloading {star}')
            try:
                delattr(self, escaped_star)
            except AttributeError:
                pass

            getattr(self, escaped_star)


    def _delete_all(self):
        removed = []
        for star in self._attributes:
            delattr(self, star)
            removed.append(star)
        for star in removed:
            self._attributes.remove(star)

    def _from_local(self, star):
        try:
            return DACERV.from_local(star, **self._kwargs)
        except ValueError:
            if self._print_errors:
                print(cl_red | f'ERROR: {star} no local data?')
            return

    def _from_DACE(self, star):
        try:
            return DACERV.from_DACE(star, **self._kwargs)
        except (KeyError, RequestException):
            if self._print_errors:
                print(cl_red | f'ERROR: {star} no data found in DACE?')
            return

    def __getattr__(self, attr):
        ignore = attr in (
            '__wrapped__',
            #   '_ipython_canary_method_should_not_exist_',
            '_repr_mimebundle_',
            'getdoc',
            '__call__',
            'items')
        ignore = ignore or attr.startswith('_repr')
        ignore = ignore or attr.startswith('_ipython')
        ignore = ignore or attr in self._ignore_attrs
        if ignore:
            return

        star = deescape(attr)
        if self.local_first:
            t = self._from_local(star)
            if t is None:
                t = self._from_DACE(star)
        else:
            t = self._from_DACE(star)

        if t is None:
            return

        setattr(self, attr, t)
        self._attributes.add(star)

        return t
