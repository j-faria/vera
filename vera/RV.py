
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

# standard library
import os
import sys
from glob import glob
import re
import pickle
import inspect
import warnings
from copy import deepcopy
from random import choice
import hashlib
from datetime import datetime, timezone
import subprocess
from collections import namedtuple
from collections.abc import Iterable
from itertools import zip_longest
import tarfile

# other packages
import numpy.core.defchararray as np_char
from scipy.stats import sigmaclip as dosigmaclip
from scipy import stats
from scipy import optimize
from astropy.time import Time
from astropy import units as u
try:
    import iCCF
    _iCCF_available = True
except ImportError:
    _iCCF_available = False

#! what a stupid hack!
import locale

class LocaleManager:
    def __init__(self, localename):
        self.name = localename

    def __enter__(self):
        self.orig = locale.setlocale(locale.LC_CTYPE)
        locale.setlocale(locale.LC_ALL, self.name)

    def __exit__(self, exc_type, exc_value, traceback):
        locale.setlocale(locale.LC_ALL, self.orig)


with LocaleManager("C"):
    from astropy.coordinates import SkyCoord
    from astropy.coordinates.name_resolve import NameResolveError

import pyexoplaneteu

try:
    from gatspy import periodic
except ImportError:
    raise ImportError('Please, pip install gatspy')

from astropy.timeseries import LombScargle

try:
    from cached_property import cached_property
except ImportError:
    raise ImportError('Please, pip install cached-property')

# local
from .query_dace import get_observations
from .stat_tools import wrms, false_alarm_level_gatspy
from .query_simbad import getIDs, getSPtype, get_bmv, get_vmag
from .binit import binRV
from .query_periods import get_JDs, get_random_times
from .gaia import secular_acceleration
from .HZ import getHZ_period
from .TESS import TESS
from .photometry import photometry
from .utils import (info, warning, error, chdir, get_function_call_str)

from .visibility import StarObsPlot, StarObsAxis
from .globals import _harps_fiber_upgrade, _technical_intervention, _ramp_up
from .utils import blue, red, redb, yel, yelb, green


class CacheSkyCoord(SkyCoord):
    @classmethod
    def from_name(cls, name, frame='icrs'):
        try:
            cached = pickle.load(open('CachedSkyCoords.pickle', 'rb'))
        except FileNotFoundError:
            cached = {}

        if name in cached:
            return cached[name]
        else:
            original = super(CacheSkyCoord, cls).from_name(name, frame)
            # keep the cached dict manageable
            n = len(cached)
            if n > 100:
                # remove a random cached target
                cached.pop(choice(list(cached.keys())))
            cached.update({name: original})
            pickle.dump(cached, open('CachedSkyCoords.pickle', 'wb'))
            return original


def are_you_sure(msg):
    print(msg)
    input()


def nospaces(s):
    return ''.join(s.split())


def change_color(color, amount=0.5):
    """
    Changes the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc, colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def isin(s, array, which=False):
    if which:
        for i, a in enumerate(array):
            if s in a:
                return i
    else:
        return any([s in a for a in array])


def label_next_to_line(x, y, label, ax, where='end', **kwargs):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if where == 'end':
        x0 = np.minimum(0.7 * (x[0] + x.ptp()), 0.7 * xlim[1])
        y0 = np.minimum(y[x > x0][0], ylim[1])
        ax.text(x0, y0, label, va='top', **kwargs)


def sine(t, p):
    return p[0] * np.sin(2 * np.pi * t / p[1] + p[2]) + p[3]


def sinefit(t, y, ye, p0, **kwargs):
    return optimize.leastsq(lambda p, t, y, ye: (sine(t, p) - y) / ye, p0,
                            args=(t, y, ye), **kwargs)



class RV:
    """ This class holds a set of RV observations """


    def __init__(self, filenames=None, star=None, ms=True, verbose=True,
                 sigmaclip=True, adjust_means=True, arrays=None, obs=None,
                 sigmaclip_sigma=3, maxerror=10, bin=False, instruments=None,
                 keep_pipeline_versions=False, mask_drs_qc0=True, tess=True,
                 remove_secular_acceleration=False):

        self.technical_intervention = _technical_intervention
        self._update_future_runs = False

        self.verbose = verbose
        self.periodogram_calculated = False
        self._periodogram_calculated_which = ''
        self._child = False
        self._atstart = True
        self._ran_treatment = False
        self._removed_planets_planetpack = False
        self._estimated_luminosity = False
        self._mask_drs_qc0 = mask_drs_qc0
        self._reproduce = []
        self._activity_indicators = set()

        self.time, self.vrad, self.svrad = [], [], []
        self.obs = []
        self.mask = []
        self.drs_qc = []
        self.instruments, self.pipelines = [], []
        self.filenames = []
        self.fwhm, self.efwhm = [], []
        self.contrast, self.econtrast = [], []
        self.rhk, self.erhk = [], []
        self.sindex, self.esindex = [], []
        self.keplerians = []
        self._added_files = []

        # photometry
        self._tess = None
        self._superWASP = None
        self._ASAS_SN = None

        if isinstance(filenames, str):
            filenames = [filenames]

        # try to guess the star
        self.star = star
        if self.star is None:
            if filenames is None:
                self.star = 'unknown star'
            else:
                self.star = filenames[0].split('_')[0]
                if self.verbose:
                    print(blue | f'Guessed star to be: ', end='')
                    print(self.star)

        if 'TOI' in self.star and tess:
            if self.verbose:
                print(
                    blue
                    | 'Found TOI in star name, looking for TESS lightcurve '
                    '(can take a while)', flush=True)
            self.search_for_TESS()


        if isinstance(instruments, str):
            instruments = [instruments]

        if filenames is None:
            assert arrays is not None, 'Need either filenames or arrays'
            self.time, self.vrad, self.svrad, self.fwhm, self.efwhm = arrays
            if obs is not None:
                self.obs = obs
            else:
                self.obs = np.ones_like(self.time, dtype=int)

            if instruments is not None:
                self.instruments = instruments
            else:
                self.instruments = ['unknown']

            self.mask = np.ones_like(self.time, dtype=bool)
            self.drs_qc = np.ones_like(self.time, dtype=bool)

            # these are not available
            self.contrast = np.full_like(self.time, np.nan)
            self.econtrast = np.full_like(self.time, np.nan)
            self.rhk = np.full_like(self.time, np.nan)
            self.erhk = np.full_like(self.time, np.nan)
            self.sindex = np.full_like(self.time, np.nan)
            self.esindex = np.full_like(self.time, np.nan)

        else:
            # alphabetical
            # filenames = sorted(filenames)
            # alphabetical, but ESPRESSO always last
            # filenames = sorted(filenames, key=lambda x: ('ESPRESSO' in x, x))
            # reverse-alphabetical, but ESPRESSO always first!
            # filenames = sorted(filenames, key=lambda x: ('ESPRESSO' in x, x),
            #                    reverse=True)
            i = 0
            for _, filename in enumerate(filenames):
                if not os.path.exists(filename):
                    raise FileNotFoundError('Cannot find file "%s"' % filename)

                # read the data from a .rdb file
                d = np.genfromtxt(filename, names=True, comments='--')

                # if verbose:
                #     print(f'From "{filename}", read: {d.dtype.names}')
                if d.size == 1:
                    if verbose:
                        msg = f'{filename} has only 1 observation!'
                        warning(msg)
                    # continue

                try:
                    # get out early if instruments were provided
                    if instruments is not None:
                        raise IndexError

                    _star_ = ''.join(self.star.split())
                    pat = re.compile(_star_ + r'_([\w\.\&-]+).rdb')
                    try:
                        instrument, pipe = pat.findall(filename)[0].split('_')
                    except ValueError:
                        instrument = pat.findall(filename)[0]
                        pipe = ''

                    if 'arXiv' in pipe or 'A&A' in pipe:
                        pipe = ''

                except IndexError:
                    if instruments is not None:
                        instrument = instruments[_]
                    else:
                        instrument = 'unknown'
                    pipe = ''
                # print(instrument, pipe)

                self.instruments.append(instrument)
                self.pipelines.append(pipe)
                self.filenames.append(filename)

                try:
                    self.time.append(np.atleast_1d(d['jdb']))
                except ValueError:
                    try:
                        self.time.append(np.atleast_1d(d['bjd']))
                    except ValueError:
                        self.time.append(np.atleast_1d(d['rjd']))

                self.vrad.append(np.atleast_1d(d['vrad']))
                self.svrad.append(np.atleast_1d(d['svrad']))

                self.obs.append(np.ones(d.size) + i)

                self.mask.append(np.ones(d.size, dtype=bool))

                if 'drs_qc' in d.dtype.names:
                    self.drs_qc.append(np.atleast_1d(d['drs_qc']))
                else:
                    self.drs_qc.append(np.ones(d.size))

                def add_array(name, zero_to_nan=False):
                    ename = 'e' + name
                    if name in d.dtype.names:
                        if zero_to_nan:
                            if np.count_nonzero(d[name]) == 0:
                                d[name] *= np.nan

                        getattr(self, name).append(np.atleast_1d(d[name]))
                        if ename in d.dtype.names:
                            getattr(self, ename).append(np.atleast_1d(d[ename]))
                        else:
                            getattr(self, ename).append(np.zeros(d.size))
                    else:
                        getattr(self, name).append(np.nan * np.ones(d.size))
                        getattr(self, ename).append(np.nan * np.ones(d.size))

                add_array('fwhm')
                add_array('contrast')
                add_array('rhk', zero_to_nan=True)
                add_array('sindex')

                i += 1

        self.instruments = np.array(self.instruments)
        self.build_arrays()
        self.units = 'km/s'

        ## read parameters, if file exists
        par_file = f'{self.star}.parameters'
        if os.path.exists(par_file):
            if self.verbose:
                print(blue | f'Reading parameters from {par_file}')
            with open(par_file) as f:
                for line in f.readlines():
                    att, val = line.split(':')
                    setattr(self, att, float(val))

        if len(self.fwhm) == 0:
            self.fwhm = np.zeros_like(self.time)
            self.efwhm = np.zeros_like(self.time)

        self._removed_secular_acceleration = False
        self._atstart = False
        self._multiple_pipeline_versions = keep_pipeline_versions

        self.did_ms = ms
        if ms:
            self.to_ms()

        self.did_sigmaclip = sigmaclip
        self.sigmaclip_sigma = sigmaclip_sigma
        self.maxerror = maxerror

        if sigmaclip:
            self.sigmaclip(sigma=sigmaclip_sigma)
            self.sigmaclip_errors(maxerror=maxerror)

        self.did_adjustmeans = adjust_means
        self._did_adjust_mean = False

        # subtract secular acceleration before calculating averages
        if remove_secular_acceleration:
            self.secular_acceleration(plot=False)

        if adjust_means:
            self.adjust_means()

        # keep a copy
        self._unbinned = {
            'time': self.time.copy(),
            'vrad': self.vrad.copy(),
            'svrad': self.svrad.copy(),
            'mask': self.mask.copy(),
            'obs': self.obs.copy(),
        }
        self.did_bin_nightly = False
        if bin:
            self.did_bin_nightly = True
            self.bin()

    ## CONSTRUCTORS
    ###############

    @classmethod
    def from_DACE(cls, star, exclude=None, ESPRESSO_only=False,
                  keep_pipeline_versions=False, **kwargs):
        v = kwargs.get('verbose', True)
        kwargs['keep_pipeline_versions'] = keep_pipeline_versions
        remove_ESPRESSO_commissioning = kwargs.pop(
            'remove_ESPRESSO_commissioning', True)

        if ESPRESSO_only:
            star, data = get_observations(
                star, instrument='ESPRESSO', rdb=True, verbose=v,
                save_versions=keep_pipeline_versions,
                remove_ESPRESSO_commissioning=remove_ESPRESSO_commissioning)

            filename = data['files']
            # filename = f"{''.join(star.split())}_ESPRESSO.rdb"

        else:
            star, data = get_observations(
                star, rdb=True, verbose=v,
                save_versions=keep_pipeline_versions,
                remove_ESPRESSO_commissioning=remove_ESPRESSO_commissioning)

            filename = data['files']
            # filename = glob(f"{''.join(star.split())}*.rdb")

            # exclude instruments by full name matching
            if exclude is not None:
                if isinstance(exclude, str):
                    exclude = (exclude, )
                filename = [
                    f for f in filename
                    if not any(excl == f.split('_')[1].split('.')[0]
                               for excl in exclude)
                ]

            # exclude instruments by part name matching
            exclude_sub = kwargs.pop('exclude_sub', None)
            if exclude_sub is not None:
                if isinstance(exclude_sub, str):
                    exclude_sub = (exclude_sub, )
                filename = [
                    f for f in filename
                    if not any(excl in f for excl in exclude_sub)
                ]

        # star name without spaces
        _star_ = nospaces(star)

        if keep_pipeline_versions:
            # read all instruments and all pipeline versions
            pass
        else:
            # read all instruments, only latest pipeline versions
            allfiles = np.array(sorted(filename, reverse=True))

            # find instrument names
            if '+' in _star_: # escape '+' in star name
                _star__ = _star_.replace('+', r'\+')
                pat = re.compile(rf'{_star__}_(\w+)_\w+')
            else:
                pat = re.compile(rf'{_star_}_(\w+)_\w+')
            insts = list(map(pat.findall, allfiles))

            unique_inst_idx = np.unique(insts, return_index=True)[1]
            filename = allfiles[unique_inst_idx]

        c = cls(filename, star=star, **kwargs)

        # store the date of the last DACE query
        time_stamp = datetime.now(timezone.utc).isoformat().split('.')[0]
        open(f'{_star_}.lastDACE.dat', 'w').write(time_stamp)
        c._lastDACEquery = time_stamp
        c._DACE_data = data

        # c._multiple_pipeline_versions = keep_pipeline_versions

        c._reproduce.append(
            get_function_call_str(inspect.currentframe(), top=True))
        return c

    @classmethod
    def from_local(cls,
                   star,
                   directory=None,
                   exclude=None,
                   ESPRESSO_only=False,
                   keep_pipeline_versions=False,
                   **kwargs):

        kwargs['keep_pipeline_versions'] = keep_pipeline_versions
        kwargs.pop('remove_ESPRESSO_commissioning', None)  # ignore

        # star name without spaces
        _star_ = nospaces(star)

        if directory is None:
            directory = '.'

        if ESPRESSO_only:
            filename = glob(f"{directory}/{_star_}_ESPRESSO*.rdb")
            _ = kwargs.pop('exclude_sub', None)
        else:
            filename = glob(f"{directory}/{_star_}*.rdb")

            # exclude instruments by full name matching
            if exclude is not None:
                if isinstance(exclude, str):
                    exclude = (exclude, )
                filename = [
                    f for f in filename
                    if not any(excl == f.split('_')[1].split('.')[0]
                               for excl in exclude)
                ]

            # exclude instruments by part name matching
            exclude_sub = kwargs.pop('exclude_sub', None)
            if exclude_sub is not None:
                if isinstance(exclude_sub, str):
                    exclude_sub = (exclude_sub, )
                filename = [
                    f for f in filename
                    if not any(excl in f for excl in exclude_sub)
                ]

        if keep_pipeline_versions:
            # read all instruments and all pipeline versions
            pass
        else:
            # read all instruments, only latest pipeline versions
            allfiles = np.array(sorted(filename, reverse=True))

            # find instrument names
            if '+' in _star_:  # escape '+' in star name
                _star_ = _star_.replace('+', r'\+')

            pat = re.compile(rf'{_star_}_([^\W_]+)')
            insts = list(map(pat.findall, allfiles))

            unique_inst_idx = np.unique(insts, return_index=True)[1]
            filename = allfiles[unique_inst_idx]

        if len(filename) == 0:
            raise ValueError(star, ': need at least one instrument to read')

        c = cls(filename, star=star, **kwargs)

        c._reproduce.append('# you might have to run `RV.from_DACE` first')
        c._reproduce.append(
            get_function_call_str(inspect.currentframe(), top=True))
        return c

    @classmethod
    def from_CCF(cls, filenames, instrument=None, overwrite=False, **kwargs):
        """ from given CCF file(s)

        filenames : str or iterable
            Filename or list of filenames
        instrument : str, optional
            Name of the instrument(s). If not given, guess from the filename
        """
        if not _iCCF_available:
            print('Please (pip) install iCCF to use this function')
            return

        verbose = kwargs.get('verbose', True)

        if isinstance(filenames, str):
            filenames = [filenames, ]

        I = iCCF.from_file(filenames)
        N = len(I)

        if N == 0:
            raise ValueError('Did not find any CCF files')

        OBJECT = I[0].HDU[0].header['OBJECT']
        star = kwargs.pop('star', None)

        instruments = np.unique([i.instrument for i in I])
        Ni = len(instruments)

        instrument_name = instrument

        if verbose:
            if star is None:
                info(f'star is {OBJECT}')
            else:
                info(f'OBJECT is {OBJECT} in fits file, renaming to {star}')

            info(f'Found {N} CCFs from {Ni} instruments')
            print(instruments)
            if instrument_name is not None:
                print(blue | '    :', f'renaming to "{instrument_name}"')

        if star is None:
            star = OBJECT

        time = np.array([i.bjd for i in I])
        vrad = np.array([i.RV for i in I])
        svrad = np.array([i.RVerror for i in I])

        fwhm = np.array([i.FWHM for i in I])
        efwhm = np.array([i.FWHMerror for i in I])

        filenames = []
        for instrument in instruments:
            # rename?
            instrument_name = instrument_name or instrument

            mask = np.array([i.instrument == instrument for i in I])

            t = time[mask]
            y = vrad[mask]
            e = svrad[mask]
            f = fwhm[mask]  # FWHM
            ef = efwhm[mask]  # np.nan * np.ones_like(t)  # error

            filename = f'{star}_{instrument}_fromCCF.rdb'
            if os.path.exists(filename) and not overwrite:
                a = input(f'File "{filename}" exists. Overwrite? (Y/n) ')
                if a.lower() == 'n':
                    print('Doing nothing')
                    return

            headertext = 'jdb\tvrad\tsvrad\tfwhm\tefwhm\n'
            headertext += '---\t----\t-----\t----\t-----'
            fmt = ['%.6f'] + 4 * ['%.9f']
            np.savetxt(filename, np.c_[t, y, e, f, ef], comments='', fmt=fmt,
                       header=headertext, delimiter='\t')

            filenames.append(filename)

        c = cls(filenames, star=star, **kwargs)

        c.cRV = iCCF.chromaticRV(I)

        c._reproduce.append(
            get_function_call_str(inspect.currentframe(), top=True))
        return c

    @classmethod
    def from_arrays(cls, time, vrad, svrad, fwhm=None, efwhm=None, **kwargs):
        if fwhm is None:
            fwhm = np.full_like(time, np.nan)
            efwhm = np.full_like(time, np.nan)
        if efwhm is None:
            efwhm = np.full_like(time, np.nan)

        mask = kwargs.pop('mask', None)
        instruments = kwargs.pop('instruments', None)


        c = cls(arrays=(time, vrad, svrad, fwhm, efwhm), **kwargs)

        if mask is not None:
            c.mask = mask
        if instruments is not None:
            c.instruments = np.atleast_1d(instruments)


        # need to test this; EDIT: it actually works, but it's ugly...
        c._reproduce.append(
            get_function_call_str(inspect.currentframe(), top=True))
        return c

    def add(self, filenames, instrument=None, skip=2, usecols=(0, 1, 2),
            units='same', adjust_mean=True):
        """ Add observations from given filename(s) to this dataset.

        filenames : str or iterable
            Filename or list of filenames
        instrument : str, optional
            Name of the instrument(s). If not given, guess from the filename
        skip : int
            Number of header lines to skip [passed to np.genfromtxt]
        usecols : tuple
            Column numbers to read [passed to np.genfromtxt]
        units : str
            If 'same', assumes the same units as the current dataset.
            OTHER OPTIONS NOT IMPLEMENTED YET
        adjust_mean : bool (default True)
            Whether to subtract the mean of the loaded radial velocities
        """

        if isinstance(filenames, str):
            if '*' in filenames:
                filenames = glob(filenames)
            else:
                filenames = [filenames, ]

        if isinstance(instrument, str) or instrument is None:
            instrument = [instrument] * len(filenames)

        if len(instrument) != len(filenames):
            ni, nf = len(instrument), len(filenames)
            msg = f'Got {nf} filenames but {ni} instruments'
            raise ValueError(msg)

        # to get around rebuilding self.each below
        self._individual_did_ms = [self.did_ms] * self.instruments.size

        for file, inst in zip(filenames, instrument):

            if file in self._added_files:
                if self.verbose:
                    print(yelb | 'Warning: ', file)
                    print('already added observations from this file!')
                continue

            if self.verbose:
                print(blue | 'Reading from', file)

            data = np.genfromtxt(file, skip_header=skip, usecols=usecols).T
            N = data.shape[1]

            if data.shape[0] < 3:
                raise ValueError(f'File {file} needs to contain at least 3 '
                                 'columns for time, RV, and uncertainty.')

            if inst is None:
                pat = re.compile(r'_(\w+)\.\w\w\w')
                match = pat.findall(file)
                if len(match) == 1:
                    inst = match[0]
                    if self.verbose:
                        print(f'Guessing instrument to be "{inst}"')

            t, y, e, *_ = data

            if units == 'same':
                self._individual_did_ms.append(not self.did_ms)
            elif units in ('m/s', 'ms') and self.units == 'km/s':
                print('not same units!')
                y *= 1e-3
                e *= 1e-3
                self._individual_did_ms.append(False)
            elif units in ('km/s', 'kms') and self.units == 'm/s':
                print('not same units!')
                y *= 1e3
                e *= 1e3
                self._individual_did_ms.append(True)

            if self.did_adjustmeans:
                if self.verbose:
                    print(yel | f'Subtracting mean from RV')
                y -= y.mean()

            self.time = np.r_[self.time, t]
            self.vrad = np.r_[self.vrad, y]
            self.svrad = np.r_[self.svrad, e]
            #
            self.fwhm = np.r_[self.fwhm, np.full(N, np.nan)]
            self.efwhm = np.r_[self.efwhm, np.full(N, np.nan)]

            #
            self.contrast = np.r_[self.contrast, np.full(N, np.nan)]
            self.econtrast = np.r_[self.econtrast, np.full(N, np.nan)]
            self.rhk = np.r_[self.rhk, np.full(N, np.nan)]
            self.erhk = np.r_[self.erhk, np.full(N, np.nan)]
            self.sindex = np.r_[self.sindex, np.full(N, np.nan)]
            self.esindex = np.r_[self.esindex, np.full(N, np.nan)]

            self.mask = np.r_[self.mask, np.full(N, True)]
            ni = self.instruments.size
            self.obs = np.r_[self.obs, (ni + 1) * np.ones(N, dtype=int)]
            self.instruments = np.r_[self.instruments, [inst]]
            self.filenames.append(file)
            self.pipelines.append('unknown')
            self.drs_qc = np.r_[self.drs_qc, np.ones(N)]

            # reset self.each
            try:
                del self.each
            except AttributeError:
                pass
            finally:
                # unit conversion is taken care already
                # old_did_ms = self.did_ms
                # self.did_ms = False
                self.each
                # self.did_ms = old_did_ms

            self._added_files.append(file)

    def add_from_CCF(self, filenames, instrument=None, adjust_mean=True):
        """ Add observations from given CCF (or .tar) file(s) to this dataset.

        filenames : str or iterable
            Filename or list of filenames
        instrument : str, optional
            Name of the instrument(s). If not given, guess from the filename
        adjust_mean : bool (default True)
            Whether to subtract the mean of the loaded radial velocities
        """
        if not _iCCF_available:
            print('Please (pip) install iCCF to use this function')
            return

        if isinstance(filenames, str):
            filenames = [filenames, ]

        if len(filenames) == 0:
            error('No files to read')
            return

        if np.char.endswith(np.array(filenames), '.tar').any():
            if self.verbose:
                info(f'Found .tar files, trying to extract CCFs')

            ccf_filenames = []
            filenames = np.array(filenames)
            mask = np.char.endswith(filenames, '.tar')
            tarfiles = filenames[mask]

            from tqdm import tqdm
            for tarf in tqdm(tarfiles):
                path_to_tar = os.path.dirname(tarf)
                tar = tarfile.open(tarf)
                for member in tar.getmembers():
                    if 'ccf_G2_A.fits' in member.name:
                        member.name = os.path.basename(member.name)
                        tar.extract(member, path=path_to_tar)
                        ccf_filenames.append(os.path.join(path_to_tar, member.name))
                tar.close()

            filenames = ccf_filenames

        I = iCCF.from_file(filenames)
        N = len(I)
        instruments = np.unique([i.instrument for i in I])
        Ni = len(instruments)

        instrument_name = instrument

        if self.verbose:
            info(f'Found {N} CCFs from {Ni} instruments')
            print(instruments)
            if instrument_name is not None:
                print(blue | '    :', f'renaming to "{instrument_name}"')

        time = np.array([i.bjd for i in I])
        vrad = np.array([i.RV for i in I])
        svrad = np.array([i.RVerror for i in I])

        for instrument in instruments:
            # rename?
            instrument_name = instrument_name or instrument

            if instrument in self.instruments and instrument_name is None:
                raise ValueError(
                    'Already have observations from this instrument!')

            # self._added_files.append(file)
            mask = np.array([i.instrument == instrument for i in I])

            t = time[mask]
            y = vrad[mask]
            e = svrad[mask]
            f = np.nan * np.ones_like(t)  # FWHM
            ef = np.nan * np.ones_like(t)  # error

            if self.units == 'm/s':
                y = (y - y.mean()) * 1e3
                e *= 1e3

            if adjust_mean:
                if self.verbose:
                    print(yel | f'Subtracting mean from RV')
                y -= y.mean()

            filename = f'{self.star}_{instrument}_fromCCF.rdb'
            if os.path.exists(filename):
                a = input(f'File "{filename}" exists. Overwrite? (Y/n)')
                if a.lower() == 'n':
                    return

            headertext = 'jdb\tvrad\tsvrad\tfwhm\tefwhm\n'
            headertext += '---\t----\t-----\t----\t-----'
            fmt = ['%.6f'] + 4 * ['%.9f']
            np.savetxt(filename, np.c_[t, y, e, f, ef], comments='', fmt=fmt,
                       header=headertext, delimiter='\t')

            self.time = np.r_[self.time, t]
            self.vrad = np.r_[self.vrad, y]
            self.svrad = np.r_[self.svrad, e]

            #
            self.fwhm = np.r_[self.fwhm, f]
            self.efwhm = np.r_[self.efwhm, ef]

            #
            self.contrast = np.r_[self.contrast, np.nan * np.ones(N)]
            self.econtrast = np.r_[self.econtrast, np.nan * np.ones(N)]
            self.rhk = np.r_[self.rhk, np.nan * np.ones(N)]
            self.erhk = np.r_[self.erhk, np.nan * np.ones(N)]
            self.sindex = np.r_[self.sindex, np.nan * np.ones(N)]
            self.esindex = np.r_[self.esindex, np.nan * np.ones(N)]

            self.mask = np.r_[self.mask, np.ones(N, dtype=bool)]
            ni = self.instruments.size
            self.obs = np.r_[self.obs, (ni+1)*np.ones(N, dtype=int)]
            self.instruments = np.r_[self.instruments, [instrument_name]]
            self.filenames.append(filename)
            self.pipelines.append('unknown')
            self.drs_qc = np.r_[self.drs_qc, np.ones(N)]

            # reset self.each
            try:
                del self.each
            except AttributeError:
                pass
            finally:
                self.each

    def change_instrument_names(self, old, new):
        if old not in self.instruments:
            error(f'No instrument named "{old}"')
            return
        if not isinstance(new, str):
            new = str(new)
        if new[0].isdigit():
            error(f"New nane '{new}' starts with a digit. That doesn't work...")
            return
        i = list(self.instruments).index(old)
        self.instruments[i] = new

        if self.verbose:
            info(f'Changed instrument name {old} --> {new}')

        try:
            del self.each
        except AttributeError:
            pass
        finally:
            self.each

    def split_instrument(self, time, new_name=None):
        from .utils import split_rdb_file_first_column
        mask = self.time > time
        index = int(self.obs[mask][0]) - 1
        inst = self.instruments[index]
        mask_this_instrument = self.each[index].time > time

        if new_name is None:
            new_name = inst + '.2'

        info(f'Splitting instrument {inst} at t={time}. '
             f'New instrument: {new_name}')

        self.instruments = np.append(self.instruments, new_name)

        old_file, ext = os.path.splitext(self.filenames[index])
        new_file = f'{old_file}_{new_name}{ext}'
        old_file_new_name = f'{old_file}_split{ext}'
        split_rdb_file_first_column(self.filenames[index], time, new_file,
                                    old_file_new_name)
        self.filenames[index] = old_file_new_name
        self.filenames.append(new_file)

        # same pipeline
        pipe = self.pipelines[index]
        self.pipelines.append(pipe)

        # add to _DACE_data
        try:
            self._DACE_data
            self._DACE_data[new_name] = {}
            self._DACE_data[new_name][pipe] = {}
            mode = list(self._DACE_data[inst][pipe])[0]
            self._DACE_data[new_name][pipe][mode] = {}
            for k in list(self._DACE_data[inst][pipe][mode].keys()):
                self._DACE_data[new_name][pipe][mode][k] = list(
                    np.array(self._DACE_data[inst][pipe][mode][k])[mask_this_instrument]
                )
                self._DACE_data[inst][pipe][mode][k] = list(
                    np.array(self._DACE_data[inst][pipe][mode][k])[~mask_this_instrument]
                )
        except AttributeError:
            pass

        # next obs
        self.obs[mask] += 1
        m = self._unbinned['time'] > time
        self._unbinned['obs'][m] += 1

        try:
            del self.each
        except AttributeError:
            pass
        finally:
            self.each

    def copy(self):
        return deepcopy(self)

    def set_mask(self, mask):
        assert isinstance(mask, np.ndarray)
        assert mask.size == self.mask.size

        self.mask = mask

        # mask the point(s) in self.each also
        for i, inst in enumerate(self.each):
            mask_inst = self.obs == i + 1
            inst.mask[:] = False
            inst.mask[mask[mask_inst]] = True

    def remove_point(self, index):
        index = np.atleast_1d(index)
        # index[index < 0] = self.NN + index[index < 0]

        instrument = self.instruments[(self.obs[index] - 1).astype(int)]
        pipeline = np.array(self.pipelines)[(self.obs[index] - 1).astype(int)]

        if self.verbose:
            word = 'point at index' if len(index) == 1 else 'points at indices'
            info(f'masking data {word} {index}')
            word = 'instrument' if len(instrument) == 1 else 'instruments'
            print(5 * ' ', f'belonging to {word} {instrument}')

        self.mask[index] = False

        # mask the point(s) in self.each also
        for ind, inst, pipe in zip(index, instrument, pipeline):
            try:
                i = getattr(self.each, inst)
            except AttributeError:
                pipe = pipe.replace('-', '_').replace('.', '_')
                i = getattr(self.each, f'{inst}_{pipe}')
            mask_out = np.where(i.time == self.time[ind])
            i.mask[mask_out] = False

    def remove_instrument(self, instrument):
        """ Remove observations from one instrument """
        if instrument not in self.instruments:
            raise ValueError(f'Cannot find "{instrument}" in instruments '
                             f'(options are {", ".join(self.instruments)})')

        i = np.where(self.instruments == instrument)[0][0]
        ind = self.obs == i + 1
        self.time = self.time[~ind]
        self.vrad = self.vrad[~ind]
        self.svrad = self.svrad[~ind]

        self.fwhm = self.fwhm[~ind]
        self.efwhm = self.efwhm[~ind]

        self.contrast = self.contrast[~ind]
        self.econtrast = self.econtrast[~ind]

        self.rhk = self.rhk[~ind]
        self.erhk = self.erhk[~ind]

        self.sindex = self.sindex[~ind]
        self.esindex = self.esindex[~ind]

        self.obs = self.obs[~ind]
        self.obs[self.obs > i+1] -= 1
        self.mask = self.mask[~ind]

        self.instruments = np.delete(self.instruments, i)
        f = self.filenames.pop(i)
        if f in self._added_files:
            self._added_files.remove(f)
        self.pipelines.pop(i)

        try:
            del self.each
        except AttributeError:
            pass
        finally:
            self.each

        if self.verbose:
            print(yel | f'Removed observations from instrument "{instrument}"')

    def unmask_all(self):
        self.mask[:] = True

    def __repr__(self):
        n1 = self.mask.sum()
        n2 = self.mask.size
        return f'RV({self.star}; {n1} (/{n2}) points)'


    ## PROPERTIES
    #############
    @property
    def help(self):
        def grouped(iterable, n):
            "s -> (s0,s1,..,sn-1), (sn,sn+1...s2n-1), ..."
            return zip_longest(*[iter(iterable)] * n)

        print(f'this object holds RV observations for {self.star}')
        print(f'from {len(self.N)} instruments: {self.instruments}')
        print('available methods:')
        cols = 3
        methods = ('plot', 'gls', 'plot_both', 'bin', 'detrend', 'save',
                   'summary', 'visibility')
        for method in grouped(methods, cols):
            for i in range(cols):
                if method[i]:
                    m = f'.{method[i]}()'
                    print(f'    {m:<15s}', end='')
            print()

        print('and attributes:')
        attrs = ('N', 'instruments', 'rms', 'error', 'datetimes')
        for attr in grouped(attrs, cols):
            for i in range(cols):
                if attr[i]:
                    a = f'.{attr[i]}'
                    print(f'    {a:<10s}', end='')
            print()

    @property
    def name(self):
        return self.star

    @cached_property
    def each(self):
        # this property is a namedtuple containing the individual instruments
        # the attributes are instances of `RV` themselves

        # don't allow for infinite recursion like s.each.INST.each.INST.each...
        if self._child:
            return

        # escape some characters before building the namedtuple
        valid_instruments = [s.replace('-', '_') for s in self.instruments]
        valid_instruments = [s.replace(' ', '_') for s in valid_instruments]
        valid_instruments = [s.replace('.', '_') for s in valid_instruments]

        if self._multiple_pipeline_versions:
            valid_pipelines = [s.replace('-', '_') for s in self.pipelines]
            valid_pipelines = [s.replace(' ', '_') for s in valid_pipelines]
            valid_pipelines = [s.replace('.', '_') for s in valid_pipelines]
            names = [i + '_' + p for i,
                     p in zip(valid_instruments, valid_pipelines)]
        else:
            names = valid_instruments

        I = namedtuple('INSTRUMENT', names)
        I.__doc__ = 'Individual instruments, accessible with `.INSTRUMENT`'
        I.__str__ = lambda a: f"INSTRUMENT({','.join(names)})"
        I.__repr__ = lambda a: f"INSTRUMENT({', '.join(names)})"
        opt = {
            'star': self.star,
            'verbose': False,
            'adjust_means': self.did_adjustmeans,
            'sigmaclip': self.did_sigmaclip,
            'maxerror': self.maxerror,
            'tess': self._tess is not None,
            'remove_secular_acceleration': self._removed_secular_acceleration,
        }

        # build the list of `RV` instances for each instrument
        i = []
        for _i, inst in enumerate(names):
            # if len(self.filenames) > 0:
            #     f = self.filenames[_i]
            #     if hasattr(self, '_individual_did_ms'):
            #         ms = self._individual_did_ms[_i]
            #         i.append(RV(f, instruments=inst, ms=ms, **opt))
            #     else:
            #         i.append(RV(f, instruments=inst, ms=self.did_ms, **opt))
            # else:
            if hasattr(self, '_individual_did_ms'):
                ms = self._individual_did_ms[_i]
            else:
                ms = self.did_ms

            m = self.obs == _i + 1
            _rv = RV.from_arrays(self.time[m], self.vrad[m], self.svrad[m],
                                 self.fwhm[m], self.efwhm[m],
                                 mask=self.mask[m], instruments=inst, ms=False,
                                 **opt)

            for ind in ('contrast', 'rhk', 'sindex'):
                setattr(_rv, ind, getattr(self, ind)[m])
                setattr(_rv, 'e' + ind, getattr(self, 'e' + ind)[m])

            i.append(_rv)

        # each instance holds a `global_mask` attribute which has the same size
        # as the "parent"'s (self) mask and is True at the indices of each
        # individual instrument
        global_mask = np.zeros_like(self.mask)
        n = 0
        for ii in i:
            ii._child = True
            ii.global_mask = global_mask.copy()
            ii.global_mask[n : n + ii.time.size] = True
            n += ii.time.size

        return I(*i)

    @property
    def rms(self):
        """ Weighted rms of the (masked) radial velocities """
        if self.mask.sum() == 0:  # only one point
            return np.nan
        else:
            return wrms(self.vrad[self.mask], self.svrad[self.mask])

    @property
    def sigma(self):
        """ Average error bar """
        if self.mask.sum() == 0:  # only one point
            return np.nan
        else:
            return self.svrad[self.mask].mean()

    error = sigma  # alias!

    @property
    def N(self):
        """ Total number of observations per instrument """
        n = {}
        for i, inst in enumerate(self.instruments):
            n[inst] = (self.obs[self.mask] == i + 1).sum()
        return n

    @property
    def NN(self):
        """ Total number of observations, all instruments together """
        return self.time[self.mask].size

    @property
    def N_complete(self):
        """ Number of observations per instrument, including masked ones """
        n = {}
        for i, inst in enumerate(self.instruments):
            n[inst] = (self.obs[self.mask] == i + 1).sum(), \
                      (self.obs[~self.mask] == i + 1).sum()
        return n

    @property
    def N_nights(self):
        return np.unique(self.time.astype(int)).size

    @property
    def points(self):
        return [(t, v, e) for t, v, e in zip(self.time, self.vrad, self.svrad)]

    @property
    def mtime(self):
        return self.time[self.mask]

    @property
    def mvrad(self):
        return self.vrad[self.mask]

    @property
    def msvrad(self):
        return self.svrad[self.mask]

    @property
    def mfwhm(self):
        return self.fwhm[self.mask]

    @property
    def mefwhm(self):
        return self.efwhm[self.mask]

    @property
    def mobs(self):
        return self.obs[self.mask]

    @cached_property
    def hash(self):
        """
        MD5 hash of `self.filenames`. Can be used as a unique data identifier.
        """
        hasher = hashlib.md5()
        for f in self.filenames:
            with open(f, 'rb') as file:
                buf = file.read()
                hasher.update(buf)
        return hasher.hexdigest()

    @property
    def reproduce(self):
        return '\n'.join(self._reproduce)

    @property
    def othernames(self):
        """ Other identifiers for this star's name """
        try:
            return getIDs(self.star, remove_space=False)
        except ValueError as e:
            if self.verbose:
                error(e)
            return []

    @property
    def vmag(self):
        """ Magnitude in the V band """
        try:
            return get_vmag(self.star)
        except ValueError as e:
            if self.verbose:
                error(e)
            return None

    @property
    def bmv(self):
        """ B-V color """
        try:
            return get_bmv(self.star)
        except ValueError as e:
            if self.verbose:
                print(red | 'ERROR:', e)
            return None

    @property
    def _instrument_labels(self):
        uniques, unique_idx, counts = np.unique(self.instruments,
                                                return_index=True,
                                                return_counts=True)
        duplicates = self.instruments[unique_idx[counts >= 2]]
        labels = []
        if len(self.pipelines) == 0:
            pipelines = self.instruments.size * ['']
        else:
            pipelines = self.pipelines

        for i, p in zip(self.instruments, pipelines):
            if i in duplicates:
                labels.append(i + f' ({p.split("-")[0]})')
            else:
                labels.append(i)
        return labels

    @property
    def activity_indicators(self):
        return self._activity_indicators

    @cached_property
    def known_planets(self):
        """
        Information about known planets for this star (from exoplanet.eu)
        """
        try:
            othernames = getIDs(self.star, remove_space=True)
        except ValueError:
            othernames = []
        if 'Proxima' in self.star:
            othernames.append('ProximaCentauri')

        data = pyexoplaneteu.get_data()
        data['star_name'] = np_char.replace(data['star_name'], 'star', '')
        data['star_name'] = np_char.replace(data['star_name'], "'s", '')
        data['star_name'] = np_char.replace(data['star_name'], ' ', '')

        data.to_numpy()
        options = [self.star, self.star.replace(' ', '')]
        possible_names = othernames + options
        possible_names += [n + 'A' for n in possible_names]
        possible_names = np.unique(possible_names)

        search = [name in data['star_name'] for name in possible_names]
        if any(search):
            # found it!
            if self.verbose:
                msg = f'Found {self.star} as planet host in exoplanet.eu'
                print(blue | msg)
            found_name = possible_names[np.nonzero(search)[0][0]]
            ind = np.where(data['star_name'] == found_name)
            f = lambda k: data[k][ind]
            keys = data.keys()
            self.known_planets_data = {
                k: v
                for k, v in zip(keys, map(f, keys))
            }
            P = f('orbital_period')
        else:
            if self.verbose:
                msg = f"Didn't find {self.star} as planet host in exoplanet.eu"
                error(msg)
            P = None

        known = namedtuple('known_planets', ['P'])
        return known(P)

    @cached_property
    def has_planets(self):
        """ Does this star have known planets? """
        return self.known_planets.P is not None

    @property
    def raw_files(self):
        try:
            self._DACE_data
            raw_files = []
            for instrument, pipeline in zip(self.instruments, self.pipelines):
                d = self._DACE_data[instrument][pipeline]
                keys = list(d.keys())[0]
                d = d[keys]
                raw_files.append([os.path.basename(f) for f in d['raw_file']])
            return np.concatenate(raw_files)
        except AttributeError:
            return None

    @property
    def datetimes(self):
        time = np.where(self.time < 24e5, self.time + 24e5, self.time)
        return np.array([
            str(dt).split('.')[0] for dt in Time(time, format='jd').datetime
        ])

    @property
    def offset_times(self):
        # ignoring overlaps for now
        if self.instruments.size == 1:
            print(red | 'ERROR:', 'Only one instrument')
        _1 = self.time[np.ediff1d(self.obs, 0, None) > 0]
        _2 = self.time[np.ediff1d(self.obs, None, 0) > 0]
        return np.mean((_1, _2), axis=0)

    @property
    def tess(self):
        if self._tess is None:
            try:
                self.search_for_TESS()
            except Exception as e:
                print(str(e))
                print(blue | 'IFNO:', 'Could not load TESS data')
        return self._tess

    @property
    def superWASP(self):
        if self._superWASP is None:
            self.search_for_superWASP()
        return self._superWASP

    @property
    def ASAS_SN(self):
        if self._ASAS_SN is None:
            self.search_for_ASAS_SN()
        return self._ASAS_SN

    @property
    def photometry(self):
        phots = []

        if self.tess:
            phots.append(self.tess)

        if self.superWASP:
            phots.append(self.superWASP)

        if self.ASAS_SN:
            phots.append(self.ASAS_SN)

        return phots


    @cached_property
    def spectral_type(self):
        """ Spectral type, from Simbad """
        return getSPtype(self.star)

    @cached_property
    def stellar_mass(self):
        """ Stellar mass """
        return 0.0

    @cached_property
    def teff(self):
        """ Stellar effective temperature """
        return 0.0

    @cached_property
    def luminosity(self):
        """ Stellar luminosity """
        return 0.0

    @cached_property
    def prot(self):
        """ Stellar rotation period """
        return 0.0

    @property
    def HZ(self):
        """
        Limits of the conservative habitable zone in days, for 1 Mearth planet.
        """
        if self.teff == 0 or self.stellar_mass == 0 or self.luminosity == 0:
            return None

        return getHZ_period(self.teff, self.stellar_mass, 1,
                            lum=self.luminosity).value

    @property
    def has_before_and_after_fibers(self):
        """
        Does the system contain ESPRESSO observations before and after the
        technical intervention?
        """
        esp = any(['ESPRESSO' in i for i in self.instruments])
        before = self.time < self.technical_intervention
        after = self.time > self.technical_intervention
        return esp and before.any() and after.any()

    @cached_property
    def number_planets(self):
        """ Number of known planets """
        if self.known_planets.P is not None:
            return (~np.isnan(self.known_planets.P)).sum()
        else:
            return 0

    @cached_property
    def sigma_ratio(self):
        if 'ESPRESSO' not in self.instruments:
            return np.nan

        try:
            sigma_ESP = np.median(self.each.ESPRESSO.svrad)
        except AttributeError:
            sigma_ESP = np.median(self.each.ESPRESSO_1_5_1_HR21.svrad)

        try:
            sigma_other = self.each.HARPS03.error
        except AttributeError:
            sigmas = [np.median(i.svrad) for i in self.each]
            sigmas.remove(sigma_ESP)
            sigma_other = np.mean(sigmas)

        return sigma_other / sigma_ESP

    @staticmethod
    def get_other_identifiers(star, **kwargs):
        """
        Query SIMBAD for other identifiers of `star`. This is a static method,
        meant to be used with the `RV` class: RV.get_other_identifiers('HD1')
        """
        rs = kwargs.pop('remove_space', False)
        an = kwargs.pop('allnames', True)
        if not isinstance(star, str) and isinstance(star, Iterable):
            ids = []
            for s in star:
                try:
                    ids.append(
                        getIDs(s, remove_space=rs, allnames=an, **kwargs))
                except ValueError:
                    continue
            return ids
        else:
            return getIDs(star, remove_space=rs, allnames=an, **kwargs)

    @property
    def is_binned(self):
        """
        Are the observations nightly binned? More specifically, this function
        tests if there is more than one observation on any given night.
        """
        # the following line does not work when two instruments have
        # observations on the same night
        # return all(np.unique(self.time.astype(int), return_counts=True)[1] == 1)
        individually_binned = []
        for i, _ in enumerate(self.instruments):
            t = self.time[self.obs == i + 1].astype(int)
            b = all(np.unique(t, return_counts=True)[1] == 1)
            individually_binned.append(b)
        return all(individually_binned)

    @property
    def night_indices(self):
        time_int = self.time.astype(int)
        return np.unique(time_int, return_inverse=True)[1] + 1

    @cached_property
    def future_runs(self):
        periods = '102,103,104,105,106'
        filename = periods.replace(',', '_')
        self._future_runs_periods = periods
        if self._update_future_runs:
            starts, ends = get_JDs(periods)
            pickle.dump((starts, ends), open(f'P{filename}.pickle', 'wb'))
            self._update_future_runs = False
        else:
            starts, ends = pickle.load(open(f'P{filename}.pickle', 'rb'))
            # starts, ends = np.empty((2,0))
            # starts, ends = np.append(np.c_[starts, ends], np.array([s,e]).T, axis=0).T
        return starts, ends

    # @cached_property
    # def _lastDACE(self):
    #     try:
    #         return open(f'{self.star}.lastDACEquery.dat').read()
    #     except:
    #         return 'unknown'

    @property
    def _longest_name(self):
        """ return the longest out of all the instrument names """
        return max(len(inst) for inst in self.instruments)

    @property
    def coords(self):
        """ Sky coordinates for this star (RA, DEC) """
        try:
            return CacheSkyCoord.from_name(self.star)
        except NameResolveError:
            msg = f'SkyCoord cannot resolve "{self.star}", try another name: '
            name = input(msg)
            return CacheSkyCoord.from_name(name)

    ## METHODS
    ##########

    def build_arrays(self):
        mapto = [
            'time', 'vrad', 'svrad', 'obs', 'mask', 'drs_qc', 'fwhm', 'efwhm',
            'contrast', 'econtrast', 'rhk', 'erhk', 'sindex', 'esindex'
        ]
        mapto = mapto + [i for i in self.activity_indicators]
        mapto = mapto + [i + '_err' for i in self.activity_indicators]

        if self.instruments.size == 1:
            for arr in mapto:
                setattr(self, arr, np.array(getattr(self, arr)).ravel())
        else:
            for arr in mapto:
                setattr(self, arr, np.concatenate(getattr(self, arr)))

        if self._mask_drs_qc0:  # mask out points where the DRS failed
            n = (self.drs_qc == 0).sum()
            if self.verbose and n > 0:
                warning(f'masking {n} points with drs_qc=0')
            self.mask[self.drs_qc == 0] = False

    def summary(self):
        cols = ('N', 'masked', 'ΔT', 'Tmin', 'Tmax', '<RV>', '<RVerr>')
        row_format = "{:>10}" + "{:>12}" * len(cols)
        print(row_format.format("", *cols))
        for i, individual in enumerate(self.each):
            m = individual.mask & self.mask[self.obs == i + 1]
            row = (
                individual.NN,
                (~m).sum(),
                round(individual.time[m].ptp(), 2),
                individual.time[m].min().round(2),
                individual.time[m].max().round(2),
                individual.vrad[m].mean().round(2),
                individual.svrad[m].mean().round(2),
            )
            print(row_format.format(individual.instruments[0], *row))

    def to_ms(self):
        self.vrad *= 1e3
        self.svrad *= 1e3

        self.fwhm *= 1e3
        self.efwhm *= 1e3

        self.units = 'm/s'
        return self

    def to_kms(self):
        self.vrad /= 1e3
        self.svrad /= 1e3

        self.fwhm /= 1e3
        self.efwhm /= 1e3

        self.units = 'km/s'
        return self

    def set_fwhm_errors(self):
        """ Sets the errors on the FWHM equal to 2x the errors on RVs """
        self.efwhm = 2 * self.svrad
        info('Setting the FWHM uncertainties to twice the RV uncertainties')

    def _destructive(self, which, changed_units=False):
        """
        When a method is destructive of the RV array, it should call this
        function at the start, rebuild the array, and then call this function
        again at the end.
        """
        if which == 'start':
            self.time, self.vrad, self.svrad = [], [], []
            self.obs = []
            self.mask = []
            self.drs_qc = []
            inds = ['fwhm', 'contrast', 'rhk', 'sindex']
            for ind in inds:
                setattr(self, ind, [])
                setattr(self, 'e' + ind, [])
            for ind in self.activity_indicators:
                setattr(self, ind, [])
                setattr(self, ind + '_err', [])

        if which == 'end':
            old_verbose = self.verbose
            self.verbose = False

            self.build_arrays()

            if changed_units and self.did_ms:
                self.to_ms()

            if self.did_sigmaclip:
                self.sigmaclip()
                self.sigmaclip_errors()

            if self.did_adjustmeans:
                self.adjust_means()

            self.verbose = old_verbose
            self.periodogram_calculated = False

            del self.each

    def adjust_mean(self):
        """ Subtract the weighted mean from the RVs and FWHM """
        if self._did_adjust_mean:
            return

        if self.verbose:
            print(yel | 'Subtracting weighted mean (from RV and FWHM)')

        m = self.mask
        v_mean = np.average(self.vrad[m], weights=1 / self.svrad[m]**2)
        self.vrad -= v_mean
        for i in self.each:
            i.vrad -= v_mean

        f_mean = np.average(self.fwhm[m], weights=1 / self.efwhm[m]**2)
        self.fwhm -= f_mean
        for i in self.each:
            i.fwhm -= v_mean

        self._did_adjust_mean = True


    def adjust_means(self):
        """
        Subtract each instrument's (weighted) mean from the RVs and FWHM
        """
        if self.verbose:
            print(yel | 'Subtracting weighted means (from RV and FWHM)')

        ln = self._longest_name
        for i, instrument in enumerate(self.instruments):
            mask = self.obs == i + 1
            if (mask & self.mask).sum() == 0:
                if self.verbose:
                    print('  ', f'{instrument:>{ln}s}', '(no observations)')
                continue

            size = self.vrad[mask & self.mask].size
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # m = self.vrad[mask & self.mask].mean()
                m = np.average(self.vrad[mask & self.mask],
                               weights=1 / self.svrad[mask & self.mask]**2)

            if self.verbose:
                msg = f'{instrument:>{ln}s} <RV> = {m:10.3f} {self.units}'
                if size == 1:
                    msg += red | f'  only 1 observation! it will be 0.0!'

            if size == 1:
                # self.mask[mask & self.mask] = False
                self.vrad[mask] = 0.0

            if size > 1:
                self.vrad[mask] -= m
                # self.each[i].vrad -= m

            # if np.isnan(self.fwhm[mask & self.mask]).any():
            #     return
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # m = self.fwhm[mask & self.mask].mean()
                m = np.average(self.fwhm[mask & self.mask],
                               weights=1 / self.efwhm[mask & self.mask]**2)
            self.fwhm[mask] -= m

        if not self._atstart:
            self._reproduce.append(
                get_function_call_str(inspect.currentframe(), top=False))

        return self

    def sigmaclip(self, plot=False, sigma=3, fwhm=False, rhk=False,
                  sindex=False):
        """ Sigma-clip the RVs, FWHM, Rhk, and Sindex """
        self.sigmaclip_sigma = sigma

        if self.instruments.size == 1:
            # sigmaclip behaves badly with only 1 point
            if self.time.size == 1:
                return
            result = dosigmaclip(self.vrad, low=sigma, high=sigma)
            n = self.vrad.size - result.clipped.size
            if self.verbose:
                s = 's' if (n == 0 or n > 1) else ''
                print(yelb | f'--> sigma-clip RVs removed {n} point' + s)

            ind = (self.vrad > result.lower) & (self.vrad < result.upper)
            self.mask[~ind] = False

            if fwhm:  # sigma-clipping FWHM too
                if np.isnan(self.fwhm).any():
                    msg = "can't sigma clip FWHM, it's nan."
                    if self.verbose:
                        print(red | 'ERROR: ' + msg)
                else:
                    result = dosigmaclip(self.fwhm, low=sigma, high=sigma)
                    n = self.fwhm.size - result.clipped.size
                    if self.verbose:
                        print(yelb | f'--> sigma-clip FWHM removed {n} point'\
                                    + ('s' if (n==0 or n>1) else ''))

                    ind = (self.fwhm > result.lower) & (self.fwhm < result.upper)
                    self.mask[~ind] = False

            # sigma-clipping R'hk too
            if rhk and self.rhk.size > 0:
                if np.isnan(self.rhk).any():
                    msg = "can't sigma clip R'hk, it's nan."
                    if self.verbose:
                        print(red | 'ERROR: ' + msg)
                else:
                    result = dosigmaclip(self.rhk, low=sigma, high=sigma)
                    ind = (self.rhk > result.lower) & (self.rhk < result.upper)
                    # ind = ind & (self.erhk > 0)
                    self.mask[~ind] = False

                    n = self.rhk.size - ind.sum()  # result.clipped.size
                    if self.verbose:
                        print(yelb | f"--> sigma-clip R'hk removed {n} points")

            # sigma-clipping S index too
            if sindex and self.sindex.size > 0:
                if np.isnan(self.sindex).any():
                    msg = f"can't sigma clip S index, it's nan."
                    if self.verbose:
                        print(red | 'ERROR: ' + msg)
                else:
                    result = dosigmaclip(self.sindex, low=sigma, high=sigma)
                    ind = (self.sindex > result.lower) & (self.sindex < result.upper)
                    ind = ind  #& (self.esindex > 0)
                    self.mask[~ind] = False

                    n = self.sindex.size - ind.sum()  # result.clipped.size
                    if self.verbose:
                        print(
                            yelb | f"--> sigma-clip S index removed {n} points")

        else:

            def removed_msg(var, n, inst):
                if n == 0:
                    pass
                elif n > 0:
                    s = 's' if n > 1 else ''
                    msg = f'--> sigma-clip {var} removed {n} point' + s
                    msg += f' for {instrument}'
                    print(yelb | msg)

            for i, instrument in enumerate(self.instruments):

                mask1 = self.obs == i + 1

                # sigmaclip behaves badly with only 1 point
                if self.time[mask1].size in (0, 1):
                    continue

                ind1 = np.where(mask1)[0]

                result = dosigmaclip(self.vrad[mask1], low=sigma, high=sigma)
                n = self.vrad[mask1].size - result.clipped.size

                if self.verbose:
                    removed_msg('RVs', n, instrument)

                l, u = result.lower, result.upper
                mask2 = (self.vrad[mask1] > l) & (self.vrad[mask1] < u)
                ind2 = ind1[~mask2]
                self.mask[ind2] = False

                if fwhm:  # sigma-clipping FWHM too
                    if np.isnan(self.fwhm[mask1]).any():
                        if self.verbose:
                            msg = f"can't sigma clip FWHM of {instrument}, it's nan."
                            print(red | 'ERROR: ' + msg)
                        continue

                    result = dosigmaclip(self.fwhm[mask1], low=sigma,
                                         high=sigma)
                    n = self.fwhm[mask1].size - result.clipped.size
                    if self.verbose:
                        removed_msg('FWHM', n, instrument)

                    l, u = result.lower, result.upper
                    mask2 = (self.fwhm[mask1] > l) & (self.fwhm[mask1] < u)
                    ind2 = ind1[~mask2]
                    self.mask[ind2] = False

                if rhk:  # sigma-clipping R'hk too (keeping only positive values)
                    if np.isnan(self.rhk[mask1]).any():
                        msg = f"can't sigma clip R'hk of {instrument}, it's nan."
                        if self.verbose:
                            print(red | 'ERROR: ' + msg)
                        continue

                    result = dosigmaclip(self.rhk[mask1], low=sigma,
                                         high=sigma)
                    l, u = result.lower, result.upper
                    mask2 = (self.rhk[mask1] > l) & (self.rhk[mask1] < u)
                    mask2 = mask2 & (self.erhk[mask1] > 0)
                    ind2 = ind1[~mask2]
                    self.mask[ind2] = False

                    n = self.rhk[mask1].size - mask2.sum()
                    if self.verbose:
                        removed_msg("R'hk", n, instrument)

                if sindex:  # sigma-clipping S index too
                    if np.isnan(self.sindex[mask1]).any():
                        msg = f"can't sigma clip S index of {instrument}, it's nan."
                        if self.verbose:
                            print(red | 'ERROR: ' + msg)
                        continue

                    result = dosigmaclip(self.sindex[mask1], low=sigma, high=sigma)
                    l, u = result.lower, result.upper
                    mask2 = (self.sindex[mask1] > l) & (self.sindex[mask1] < u)
                    mask2 = mask2 & (self.esindex[mask1] > 0)
                    ind2 = ind1[~mask2]
                    self.mask[ind2] = False

                    n = self.sindex[mask1].size - mask2.sum()
                    if self.verbose:
                        removed_msg('S-index', n, instrument)

        return self

    def sigmaclip_errors(self, maxerror=10, plot=False, positive=True, rhk=True):
        """ Mask out points with RV error larger than `maxerror` """
        self.maxerror = maxerror
        m = self.mask
        errormask = np.abs(self.svrad) < maxerror
        if positive:
            errormask = errormask & (self.svrad > 0)

        # if rhk and positive:
        #     try:
        #         self.erhk
        #         errormask = errormask & (self.erhk > 0)
        #     except AttributeError:
        #         pass

        n = self.time[m].size - self.time[m & errormask].size
        if n > 0 and self.verbose:
            if positive:
                msg = f'--> sigma-clip-errors (> 0, maxerror = {maxerror} m/s)'
            else:
                msg = f'--> sigma-clip-errors (maxerror = {maxerror} m/s)'
            msg += f' removed {n} points:\n'

            print(yelb | msg, end='')

            # from which instruments did we remove, and how many?
            inds, counts = np.unique(self.obs[m & ~errormask],
                                     return_counts=True)
            inds = inds.astype(int)
            removed_from_inst = np.in1d(self.obs[m], inds)
            _, Ns = np.unique(self.obs[m][removed_from_inst],
                              return_counts=True)
            percents = 100 * counts / Ns
            howmany = list(zip(self.instruments[inds - 1], counts, percents))
            msg = '   '
            msg += ' '.join(f'{s[0]} ({s[1]}, {s[2]:.2f}%)' for s in howmany)
            print(msg)

        self.mask[~errormask] = False

    def bin(self, _child=False):
        """ Bin the data per night """
        if self.is_binned:
            self.did_bin_nightly = False
            if self.verbose:
                print(blue | 'Observations already binned')
            return

        if self.verbose:
            print(yelb | 'Binning the observations nightly')

        mask = self.mask
        obs = self.obs.copy()
        drs_qc = self.drs_qc.copy()
        time, vrad, svrad = map(np.copy, (self.time, self.vrad, self.svrad))

        ind = ['fwhm', 'contrast', 'sindex', 'rhk']
        indicator_copies = {}
        for i in ind:
            indicator_copies[i] = tuple(map(
                np.copy, (getattr(self, i), getattr(self, 'e' + i))
            ))

        indicator_copies2 = {}
        for i in self.activity_indicators:
            indicator_copies2[i] = tuple(map(
                np.copy, (getattr(self, i), getattr(self, i + '_err'))
            ))

        self._destructive('start')

        def bin_indicator(t, m, ind, a, b, pre='', post=''):
            # print(ind, np.isnan(a[m]).any(), np.isnan(b[m]).any())
            if np.isnan(a[m]).any():
                getattr(self, ind).append(np.full_like(t, np.nan))
                getattr(self, pre + ind + post).append(np.full_like(t, np.nan))
            elif np.isnan(b[m]).any():
                with warnings.catch_warnings():
                    warnings.simplefilter("error", category=RuntimeWarning)
                    _, a = binRV(time[m], a[m], None, stat='mean', tstat='mean')
                getattr(self, ind).append(a)
                getattr(self, pre + ind + post).append(np.full_like(t, np.nan))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", category=RuntimeWarning)
                    _, a, b = binRV(time[m], a[m], b[m], estat='addquad')
                getattr(self, ind).append(a)
                getattr(self, pre + ind + post).append(b)

        for i in range(self.instruments.size):
            # print(self.instruments[i])
            mobs = obs == i + 1
            m = mobs & mask

            # if self.each[i].is_binned:
            #     # t, v, e = time[mobs], vrad[mobs], svrad[mobs]
            #     t, v, e = binRV(time[mobs], vrad[mobs], svrad[mobs], estat='mean')
            # else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=RuntimeWarning)
                t, v, e = binRV(time[m], vrad[m], svrad[m], estat='addquad')

            # except OverflowError:
            #     print(red | f'Cannot bin "{self.instruments[i]}"')
            #     self._destructive('end')
            #     continue

            self.time.append(t)
            self.vrad.append(v)
            self.svrad.append(e)

            self.obs.append(np.ones_like(t, dtype=int) + i)
            self.mask.append(np.ones_like(t, dtype=bool))

            ind = binRV(time[m], vrad[m], binning_indices=True)
            self.drs_qc.append(drs_qc[m][ind])

            for ind, (a, b) in indicator_copies.items():
                bin_indicator(t, m, ind, a, b, pre='e')

            for ind, (a, b) in indicator_copies2.items():
                bin_indicator(t, m, ind, a, b, post='_err')

            # R'hk
            # if (erhk[m] == 0.0).any():
            #     erhk[m][erhk[m] == 0.0] = 1.0

        self._destructive('end')

        if not _child:
            # bin each of the individual instruments' RV instances
            for i in self.each:
                i.did_sigmaclip = self.did_sigmaclip
                i.did_adjustmeans = self.did_adjustmeans
                i.bin(_child=True)

        if not self._atstart:
            self._reproduce.append(
                get_function_call_str(inspect.currentframe(), top=False))

    def remove_instrument(self, instrument):
        if instrument not in self.instruments:
            if self.verbose:
                print(red | 'ERROR: no data from instrument "{instrument}"')
            return

        raise NotImplementedError
        # mask = self.mask
        # obs = self.obs.copy()
        # time, vrad, svrad = self.time.copy(), self.vrad.copy(), self.svrad.copy()
        # fwhm, efwhm = self.fwhm.copy(), self.efwhm.copy()
        # rhk, erhk = self.rhk.copy(), self.erhk.copy()
        # self._destructive('start')

        # self._destructive('end')

    def merge(self, keep=0):
        """ Merge RV points obtained on the same BJD """
        u, inv, c = np.unique(self.time[self.mask], return_inverse=True, return_counts=True)
        rep = c > 1
        is_repeated = np.isin(self.time[self.mask], u[rep])

        if rep.any():
            n = rep.sum()
            # find between which instruments there are repeated times
            rep_insts_id = np.unique(self.obs[self.mask][is_repeated])
            rep_insts_ind = (rep_insts_id - 1).astype(int)
            rep_insts = self.instruments[rep_insts_ind]

            if self.verbose:
                print(blue | 'INFO: ', f'Found {n} repeated times')
                print(u[rep])
                print(blue | 'INFO: ', f'Between instruments: {rep_insts}')
        else:
            if self.verbose:
                print(blue | 'INFO: ', 'All times are unique')
            return

        if self.verbose:
            info(f'Keeping observations from {rep_insts[keep]}')

        for i, time in enumerate(self.time):
            if not is_repeated[i]:
                continue
            if self.obs[i] != rep_insts_id[keep]:
                self.mask[i] = False
                # mask the correct time in the .each mask as well
                take = int(not keep)
                e = getattr(self.each, rep_insts[take])
                ind = np.where(time == e.time)[0]
                e.mask[ind] = False

    def split_HARPS_fibers(self, change_to_pre=True):
        """ Split HARPS observations before and after the fiber change, which
        occurred at MJD 57170. Two new instruments are created, with "pre" and
        "post" appended to their names.
        """
        # assert 'HARPS' in self.instruments, f'Cannot find "HARPS" in instruments'
        old = self.instruments[0]
        if 'HARPSpost' not in self.instruments:
            self.split_instrument(_harps_fiber_upgrade, 'HARPSpost')
            if change_to_pre:
                self.change_instrument_names(old, 'HARPSpre')

    def split_ESPRESSO_fibers(self):
        """ Split ESPRESSO observations before and after the ramp-up """
        if 'ESPRESSO21' not in self.instruments:
            self.split_instrument(_ramp_up, 'ESPRESSO21')


    def extract_from_DACE_data(self, quantity=None, string=False):
        """
        Get timeseries of `quantity` from self._DACE_data and set it as an
        attribute. For example, "naindex", "haindex", etc. If `quantity` is
        None, try to extract some default activity indicators.
        """
        assert hasattr(self, '_DACE_data'), "Can't find self._DACE_data"

        if quantity is None:
            quantity = ('berv', 'bispan', 'caindex', 'naindex', 'haindex',
                        'ccf_asym')
        if isinstance(quantity, str):
            quantity = (quantity, )

        if all([hasattr(self, q) for q in quantity]):
            if self.verbose:
                info('all quantities alread extracted')
            return

        for quant in quantity:
            q = []
            eq = []
            for i, p in zip(self.instruments, self.pipelines):
                mode = list(self._DACE_data[i][p].keys())[0]
                # if the quantity is in the dict keys
                if quant in self._DACE_data[i][p][mode]:
                    # append the value
                    dtype = str if string else float
                    value = np.array(self._DACE_data[i][p][mode][quant], dtype=dtype)
                    q.append(value)
                    try:
                        # try appending the errors
                        value = np.array(self._DACE_data[i][p][mode][quant + '_err'], dtype=float)
                        eq.append(value)
                    except KeyError:
                        pass
                else:
                    print(red | 'ERROR: ', end='')
                    print(f'Cannot find "{quant}" in _DACE_data')

            q = np.concatenate(q)

            errors = False
            if len(eq) > 0:
                eq = np.concatenate(eq)
                if 'NaN' not in eq:
                    errors = True

            if self.did_bin_nightly:
                if errors:
                    _, q, eq = binRV(self._unbinned['time'], q, eq)
                else:
                    _, q, _ = binRV(self._unbinned['time'], q, np.ones_like(q),
                                 stat='mean', tstat='mean')

            setattr(self, quant, q)
            self._activity_indicators.add(quant)
            if errors:
                setattr(self, quant + '_err', eq)
            else:
                setattr(self, quant + '_err', np.full_like(q, np.nan))

            for i, individual in enumerate(self.each):
                m = self.obs == i + 1
                setattr(individual, quant, q[m])
                if errors:
                    setattr(individual, quant + '_err', eq[m])
                else:
                    setattr(individual, quant + '_err', np.full_like(q[m], np.nan))

            if self.verbose:
                if errors:
                    info(f'Adding {quant} and e{quant} as attributes')
                else:
                    info(f'Adding {quant} as attribute')


    def estimate_luminosity(self):
        """
        Estimate the luminosity using a mass-luminosity relation.
        See wikipedia: https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
        """
        m = self.stellar_mass
        if m == 0:
            return 0

        if self.verbose:
            print(yelb
                  | 'Estimating luminosity from mass-luminosity relation!')
        self._estimated_luminosity = True

        if m < 0.43:
            return 0.23 * m**2.3
        elif m < 2:
            return m**4
        elif m < 20:
            return 1.4 * m**3.5
        else:
            return 32000 * m

    def visibility(self, ax=None, save=None, show=True):
        target = {'name': self.star, 'coord': self.coords}
        if ax is None:
            fig = StarObsPlot(2019, (target, ), observatory='esoparanal')
            ax = fig.axes[0]
            now = Time.now().datetime.timetuple().tm_yday
            ax.vlines(now, 0, 91, color='b')
        else:
            StarObsAxis(ax, 2019, (target, ), observatory='esoparanal')

        # if len(self.TESS_sectors) > 0:
        #     jd2019 = Time('2019-01-01').jd
        #     cycleJD = pickle.load(open('TESS_sectors_JD.pickle', 'rb'))
        #     for s in self.TESS_sectors:
        #         l1 = cycleJD[s][0] - jd2019
        #         l2 = cycleJD[s][1] - jd2019
        #         ax.axvspan(l1, l2, color='g', alpha=0.2, ec=None)

        # if not show:
        #     return fig
        # else:
        #     plt.show()

        if save:
            fig.savefig(save)

        # return fig

    def gls(self, which='vrad', recompute=False, plot=True, ax=None, FAP=True,
            adjust_offsets=True, frequency=False, plow=None, bootstrap=True,
            HZ=False, gatspy=False, legend=True, oversampling=20,
            plot_data_with_offsets=False, color=None, line_kwargs={}, **kwargs):
        """
        Calculate the Lomb-Scargle periodogram of the RVs. This function can
        automatically adjust RV offsets (for different instruments and between
        ESPRESSO fibers) while calculating the periodogram, but this is slower.
        Turn this off by setting `adjust_offsets` to False.
        """

        if self.time[self.mask].size < 3:
            print(red | 'Cannot calculate periodogram! Too few points?')
            return

        freq = self.frequency_grid()
        period = 1 / freq

        available = {
            'vrad': (self.vrad, self.svrad),
            'fwhm': (self.fwhm, self.efwhm),
            'rhk': (self.rhk, self.erhk),
            'contrast': (self.contrast, self.econtrast),
        }

        y, e = available[which]

        same = self._periodogram_calculated_which == which
        try:
            # didn't adjust offsets before but now want to do it
            if adjust_offsets and not self.GLS['gatspy']:
                recompute = True
            # adjusted offsets before but now don't want to do it
            if not adjust_offsets and self.GLS['gatspy']:
                recompute = True
        except AttributeError:
            pass

        can_adjust_offsets = self.instruments.size > 1 or self.has_before_and_after_fibers
        if not can_adjust_offsets:
            adjust_offsets = False

        if (not self.periodogram_calculated) or recompute or (not same):
            # use non-masked points
            m = self.mask
            # and not those which are nan
            m &= ~np.isnan(y)

            can_adjust_offsets = self.instruments.size > 1 or self.has_before_and_after_fibers
            if adjust_offsets and can_adjust_offsets:
                if self.verbose:
                    info('Adjust RV offsets within periodogram')

                gatspy = True
                model = periodic.LombScargleMultiband(Nterms_base=1,
                                                      Nterms_band=0)
                obs = self.obs

                model.fit(self.time[m], y[m], e[m], filts=obs[m])
                # period, power = model.periodogram_auto(oversampling=30)
                power = model.periodogram(period)
            else:
                if gatspy:
                    # if self.time.size < 50:
                    model = periodic.LombScargle(fit_period=False)
                    # else:
                    #     model = periodic.LombScargleFast()

                    model.fit(self.time[m], y[m], e[m])
                    # period, power = model.periodogram_auto(oversampling=30)
                    power = model.periodogram(period)

                else:
                    model = LombScargle(self.time[m], y[m], e[m])
                    power = model.power(1 / period)

            # save it
            self.GLS = {}
            self.GLS['model'] = model
            self.GLS['period'] = period
            self.GLS['power'] = power
            self.periodogram_calculated = True
            self._periodogram_calculated_which = which
            self.GLS['gatspy'] = gatspy
            if gatspy:
                fal = partial(false_alarm_level_gatspy, self)
                self.GLS['model'].false_alarm_level = fal

        if not self.GLS['gatspy']:
            adjust_offsets = False
            plot_data_with_offsets = False

        if self.verbose and adjust_offsets:
            print(blue | 'Adjusted means:')
            ln = self._longest_name
            offsets = self.GLS['model'].ymean_by_filt_
            instruments = self.instruments.copy().astype('U16')

            # if self.has_before_and_after_fibers:
            #     # print('(note ESPRESSO offset is between before and after fiber change)')
            #     i = np.where(instruments == 'ESPRESSO')[0][0]
            #     instruments[i] = 'ESPRESSO-post'
            #     instruments = np.insert(instruments, i, 'ESPRESSO-pre')
            #     ln += 6

            s = [
                f'  {i:{ln}s}: {off:7.4f} {self.units}'
                for i, off in zip(instruments, offsets)
            ]
            print('\n'.join(s))

        if not plot:
            return

        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
        else:
            ax = ax
            fig = ax.figure

        kw = dict(color=color, **line_kwargs)

        if frequency:
            factor = 1 #/ 86400
            ax.plot(factor / self.GLS['period'], self.GLS['power'], **kw)
        else:
            ax.semilogx(self.GLS['period'], self.GLS['power'], **kw)

        if FAP and self.time[self.mask].size > 5:
            if bootstrap:
                if self.verbose:
                    print(blue | 'calculating FAP with bootstrap...')
                k = dict(method='bootstrap')
                fap01 = self.GLS['model'].false_alarm_level(0.1, **k)
                fap001 = self.GLS['model'].false_alarm_level(0.01, **k)
            else:
                fap01 = self.GLS['model'].false_alarm_level(0.1)
                fap001 = self.GLS['model'].false_alarm_level(0.01)

            fap_period = kwargs.get('fap_period', 0.98 * ax.get_xlim()[1])
            for fap, fapstr in zip((fap01, fap001), ('10%', '1%')):
                ax.axhline(fap, color='k', alpha=0.3)
                ax.text(fap_period, fap, fapstr, ha='right',
                        va='bottom', fontsize=8, alpha=0.4)

        show_planets = kwargs.get('show_planets', True)
        if show_planets and self.known_planets.P is not None:
            # legend = legend & True
            y1, y2 = ax.get_ylim()
            h = 0.1 * abs(y2 - y1)
            P = 1 / self.known_planets.P if frequency else self.known_planets.P
            ax.vlines(P, ymin=y2 - h, ymax=y2, color='m', alpha=0.6,
                      label='planets')

        show_prot = kwargs.get('show_prot', True)
        if show_prot and self.prot:
            if isinstance(self.prot, tuple): # assume it's (prot, error)
                y1, y2 = ax.get_ylim()
                h = 0.05 * abs(y2 - y1)
                kw = dict(fmt='o', color='r', alpha=0.6, lw=2, ms=2)
                prot = 1 / self.prot[0] if frequency else self.prot[0]
                ax.errorbar(x=prot, y=y2, **kw, label=r'P$_{\rm rot}$')
                # ax.vlines(self.prot[0], ymin=y2 - h, ymax=y2, **kw)
                # ax.plot(self.prot[0], y2, 'x', **kw, label=r'P$_{\rm rot}$')

            elif not np.isnan(self.prot):
                y1, y2 = ax.get_ylim()
                h = 0.05 * abs(y2 - y1)
                kw = dict(color='r', alpha=0.6, lw=2)
                prot = 1 / self.prot if frequency else self.prot
                ax.vlines(prot, ymin=y2 - h, ymax=y2, **kw)
                ax.plot(prot, y2, 'x', **kw, label=r'P$_{\rm rot}$')

        if HZ and self.HZ is not None:
            ax.axvspan(*self.HZ, color='g', alpha=0.2, zorder=-1, label='HZ')

        xlabel = 'Frequency [days$^{-1}$]' if frequency else 'Period [days]'
        ax.set(xlabel=xlabel,
               ylabel='Normalised Power',
               ylim=(0, None),
               xlim=(1e-10, 1) if frequency else (1, None)
        )

        labels = [line.get_label() for line in ax.lines]
        # print(labels)
        # labels = not all([l.startswith('_') for l in labels])
        if legend and labels:
            fig.legend(ncol=10, bbox_to_anchor=(1, 0.9), loc='lower right',
                       fontsize='small')#, handletextpad=0.3)

        # axf.set(xlabel='Frequency [days$^{-1}$]', ylabel='Power')

        add_period_axis = kwargs.get('add_period_axis', True)
        if frequency and add_period_axis:
            f2P = lambda f: 1 / (f + 1e-10)
            P2f = lambda P: 1 / (P + 1e-10)
            ax2 = ax.secondary_xaxis("top", functions=(f2P, P2f))
            ax.xaxis.set_ticks_position('bottom')
            # ax2.minorticks_off()
            print(ax.get_xticklabels())
            ax2.set_xticks([1, 5, 10, 50, 100])
            ax2.set_xticklabels(['$1$', '$5$', '$10$', '$50$', '$100$'], rotation=45)
            ax2.set_xticks(list(range(1, 10)) + list(range(10, 100, 10)), minor=True)
            ax2.xaxis.set_ticks_position('top')
            #
            # from matplotlib.ticker import AutoMinorLocator
            # minor_locator = AutoMinorLocator(2)
            # ax.xaxis.set_minor_locator(minor_locator)
            # plt.grid(which='minor')
            #
            # ax.set_xticklabels(np.arange(0, 1.1, 0.2).round(1))
            ax.set_xlim(0, 1)
            ax2.set_xlabel('Period [days]')

        #? may want to plot the best-fit offsets together with the data
        if plot and plot_data_with_offsets and adjust_offsets:
            if which == 'vrad':
                axp, _ = self.plot()
            elif which == 'fwhm':
                axp = self.plot_fwhm()

            m = self.GLS['model']
            for filt, ym in zip(np.unique(m.filts), m.ymean_by_filt_):
                mask = m.filts == filt
                axp.hlines(ym, m.t[mask].min(), m.t[mask].max(), ls='--')

        # #? should we correct for the offset here? doesn't seem like the job for
        # #? a gls() function...
        # try:
        #     self._gls_removed_offset
        # except AttributeError:
        #     self._gls_removed_offset = False

        # if correct_offset and not self._gls_removed_offset and adjust_offsets:
        #     m = self.GLS['model']
        #     if self.verbose:
        #         of = np.ediff1d(m.ymean_by_filt_)[0]
        #         print(blue | f'Removing offset of {of} {self.units} from RVs!')

        #     self._gls_removed_offset = True
        #     for filt, ym in zip(np.unique(m.filts), m.ymean_by_filt_):
        #         mask = m.filts == filt
        #         self.vrad[mask] -= ym

        return ax

    periodogram = gls  # alias!

    def gls_both(self, contrast=False, rhk=False, sindex=False, naindex=False,
                 haindex=False, ccf_asym=False, **kwargs):
        """ Plot GLS of RVs together with that of the FWHM, or others """

        fig, (ax, ax1) = plt.subplots(ncols=1, nrows=2,
                                      constrained_layout=True, sharex=True)

        kwargs.setdefault('legend', False)

        self.gls(ax=ax, **kwargs)
        fig.legend(ncol=10, bbox_to_anchor=(1, 0.94), loc='lower right',
                   fontsize='small')#, handletextpad=0.3)

        if contrast:
            which = 'contrast'
        elif rhk:
            which = 'rhk'
        else:
            which = 'fwhm'

        self.gls(which, ax=ax1, **kwargs)

        ax.set(xlabel='')
        ax.set_title('RVs', loc='left')
        ax1.set_title(which, loc='left')
        # ax1.legend().remove()
        # ax1.set(xlabel='Time [days]')
        # if contrast:
        #     ax1.set_ylabel('Contrast [%]')
        # elif rhk:
        #     ax1.set_ylabel("R'hk")
        # elif sindex:
        #     ax1.set_ylabel("S index")
        # else:
        #     ax1.set_ylabel(f'FWHM [{self.units}]')

        # if show_fibers:
        #     ax.axvline(self.technical_intervention, color='k', ls='--', lw=2)
        #     ax1.axvline(self.technical_intervention, color='k', ls='--', lw=2)

        # ax.legend()
        return fig, (ax, ax1)

    from ._periodograms import (window_function, gls_indicator, gls_offset,
                                gls_fwhm, gls_rhk, gls_bis, gls_contrast,
                                gls_haindex, gls_naindex, gls_caindex,
                                frequency_grid)

    def gls_all(self, select=None):
        try:
            self.extract_from_DACE_data()
        except:
            pass
        try:
            self.actin()
        except:
            pass

        if select is None:
            ind = ['fwhm', 'rhk', 'sindex']
            ind = ind + list(self.activity_indicators)
            for i in ind:
                try:
                    getattr(self, i)
                except AttributeError:
                    ind.remove(i)
        else:
            ind = list(select)

        known_labels = dict(
            fwhm = f'FWHM [{self.units}]',
            rhk ='R`$_{HK}$',
            naindex = 'Na index',
            haindex = r'H$\alpha$ index',
        )

        ni = len(ind)
        fig, axs = plt.subplots(ncols=1, nrows=ni+1, constrained_layout=True,
                                sharex=True)

        instrument_labels = self._instrument_labels
        gls_kw = dict(
            # recompute=True,
            show_prot=True,
            bootstrap=False,
            show_planets=True,
        )

        self.gls(ax=axs[0], **gls_kw)
        axs[0].set_xlabel('')

        for i, w in enumerate(ind):
            try:
                self.gls_indicator(w, w + '_err', known_labels.get(w, w),
                                    ax=axs[i+1], recompute=True, **gls_kw)
            except AttributeError:
                self.gls_indicator(w, 'e' + w, known_labels.get(w, w),
                                   ax=axs[i+1], **gls_kw)

            axs[i+1].set(xlabel='', ylabel='')


        # if show_fibers:
        #     ax.axvline(self.technical_intervention, color='k', ls='--', lw=2)
        #     ax1.axvline(self.technical_intervention, color='k', ls='--', lw=2)

        # ax.legend()
        return fig, axs

    def plot_and_gls(self,
                     value='vrad',
                     error=None,
                     label=None,
                     plotkw={},
                     glskw={}):
        fig, axs = plt.subplots(2, 1, constrained_layout=True)

        guesses = {
            'vrad': ('svrad', 'RV'),
            'fwhm': ('efwhm', 'FWHM'),
            'contrast': ('econtrast', 'CCF contrast'),
            'rhk': ('erhk', r"$\log R'_{HK}$"),
            'naindex': ('enaindex', 'Na index'),
            'haindex': ('ehaindex', r'H$\alpha$ index'),
        }

        if error is None or label is None:
            error, label = guesses[value]

        plotkw.setdefault('right_ticks', False)
        plotkw.setdefault('ms', 2)
        self.plot_indicator(value, error, label, ax=axs[0], **plotkw)

        glskw.setdefault('show_title', False)
        self.gls_indicator(value, error, label, ax=axs[1], **glskw)

        return fig, axs


    def phase_fold(self, period, which='vrad'):

        available = {
            'vrad': (self.vrad[self.mask], self.svrad[self.mask]),
            'fwhm': (self.fwhm[self.mask], self.efwhm[self.mask]),
            'rhk': (self.rhk[self.mask], self.erhk[self.mask]),
            'contrast': (self.contrast[self.mask], self.econtrast[self.mask]),
        }
        if which not in available.keys():
            print('Available quantities:', list(available.keys()))
            return

        t = self.time[self.mask].copy()
        tt = np.linspace(t[0], t[-1], 1000)
        y, e = map(np.copy, available[which])

        if not self.periodogram_calculated:
            self.gls(plot=False, which=which)

        m = self.GLS['model']

        if self.GLS['gatspy']:
            filts = np.digitize(tt, np.r_[t[0], self.offset_times])
            predtt = m.predict(tt, filts, period)
            pred = m.predict(t, m.filts, period)

        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True,
                                       figsize=(8, 4))
        plot = {
            'vrad': self.plot,
            'fwhm': self.plot_fwhm,
        }[which]
        plot(ax=ax1, right_ticks=False)
        ax1.plot(tt, predtt, 'k', alpha=0.5),


        t0 = 0.0
        phase = ((t - t0) / period) % 1.0 - 0.3

        all_lines = []
        for i in np.unique(self.obs):
            ind = self.obs[self.mask] == i
            lines, caplines, barlines = ax2.errorbar(
                np.sort(phase[ind]),
                y[ind][np.argsort(phase[ind])],
                e[ind][np.argsort(phase[ind])],
                fmt='o'
            )
            all_lines.append(lines)

        # if True:
        #     from mpldatacursor import datacursor

        #     def tooltip_formatter(**kwargs):
        #         x, y = kwargs['x'], kwargs['y']
        #         ind = np.where(phase == x)[0][0]
        #         print(x, y, ind)
        #         # e = self.svrad[ind]
        #         # i = self.instruments[int(self.obs[ind]) - 1]
        #         # dt = self.datetimes[ind]
        #         # tt = f'⊢ {dt} ⊣\n'
        #         # tt += 'BJD: ' + f'{x:.5f}' + '\n'
        #         # tt += f'RV[{self.units}]: ' + f'{y:.4f} ± {e:.4f}'
        #         # return tt

        #     datacursor(all_lines, bbox=dict(fc='white', alpha=0.8), #usetex=True,
        #                 formatter=tooltip_formatter, ha='left', fontsize=8)


        # ax2.errorbar(np.sort(phase), y[np.argsort(phase)], e[np.argsort(phase)],
        #              fmt='.', color='k')


    from ._plots import (plot, plot_both, plot_indicator, plot_fwhm,
                         plot_contrast, plot_sindex, plot_rhk, plot_berv)

    def plot_all(self, include_sigma_clip=False, select=None,
                 show_fibers=False, errorbar_kwargs=None):
        """ Plot RVs together with other indicators """

        try:
            self.extract_from_DACE_data()
        except:
            pass
        try:
            self.actin()
        except:
            pass

        if select is None:
            ind = ['fwhm', 'rhk', 'sindex']
            ind = ind + list(self.activity_indicators)
            for i in ind:
                try:
                    getattr(self, i)
                except AttributeError:
                    ind.remove(i)
        else:
            ind = list(select)

        known_labels = dict(
            fwhm = f'FWHM [{self.units}]',
            rhk ='R`$_{HK}$',
            naindex = 'Na index',
            haindex = r'H$\alpha$ index',
        )

        ni = len(ind)
        fig, axs = plt.subplots(ncols=1, nrows=ni+1, constrained_layout=True,
                                sharex=True)

        instrument_labels = self._instrument_labels
        plot_kw = dict(
            include_sigma_clip=include_sigma_clip,
            show_fibers=show_fibers,
            right_ticks=False,
            legend=False,
            ms=2
        )

        self.plot(ax=axs[0], **plot_kw)
        axs[0].set_xlabel('')

        if errorbar_kwargs is None:
            errorbar_kwargs = {}
        errorbar_kwargs.setdefault('fmt', 'o')
        errorbar_kwargs.setdefault('capsize', 0)
        errorbar_kwargs.setdefault('ms', 2)

        for i, w in enumerate(ind):
            if w == 'fwhm':
                self.plot_fwhm(ax=axs[i+1], **plot_kw)
            elif w == 'rhk':
                self.plot_rhk(ax=axs[i+1], **plot_kw)
            else:
                try:
                    self.plot_indicator(w, w + '_err', known_labels.get(w, w),
                                        ax=axs[i+1], **plot_kw)
                except:
                    pass
            axs[i+1].set(xlabel='')


        # if show_fibers:
        #     ax.axvline(self.technical_intervention, color='k', ls='--', lw=2)
        #     ax1.axvline(self.technical_intervention, color='k', ls='--', lw=2)

        # ax.legend()
        return fig, axs

    def detrend(self, degree=1, quantity='vrad', plot=True, weigh=True):
        """
        Remove a trend of a given degree from the RVs

        Arguments
        ---------
        degree: int or iterable
            Trend degree, either same for all instruments or per instrument
        quantity: str, optional, default 'vrad'
            Which timeseries to detrend
        plot: bool
            Show a plot with the trend fit
        """

        m = self.mask
        t = self.time
        if quantity == 'vrad':
            v, e = self.vrad, self.svrad
        else:
            v = getattr(self, quantity)
            e = getattr(self, 'e' + quantity)

        if self.verbose:
            if quantity == 'vrad':
                print(f'Removing trend from RVs')
            else:
                print(f'Removing trend from {quantity}')

        if weigh:
            fitp = np.polyfit(t[m], v[m], degree, w=1 / e[m])
        else:
            fitp = np.polyfit(t[m], v[m], degree)

        if self.verbose:
            print(blue | f'coefficients: {fitp}')

        if plot:
            if quantity == 'vrad':
                ax, _ = self.plot()
            elif quantity == 'fwhm':
                ax, _ = self.plot_fwhm()
            elif quantity == 'rhk':
                ax = self.plot_rhk()

            tt = np.linspace(t.min(), t.max(), 1000)
            # max_zorder = max([l.get_zorder() for l in ax.get_lines()])
            ax.plot(tt, np.polyval(fitp, tt), color='k', lw=3, zorder=3)

        if quantity == 'vrad':
            self.vrad = self.vrad - np.polyval(fitp, self.time)
        else:
            setattr(self, quantity,
                    getattr(self, quantity) - np.polyval(fitp, self.time))

        self.periodogram_calculated = False
        self._reproduce.append(
            get_function_call_str(inspect.currentframe(), top=False))

    def detrend_individually(self, degree=1, plot=True, slopes=None):
        """
        Remove a trend of a given degree from the RVs, for each isntrument
        individually

        Arguments
        ---------
        degree: int or iterable
            Trend degree, either same for all instruments or per instrument
        plot: bool
            Show a plot with the trend fit
        """
        if self.verbose:
            print('Removing trend from RVs (for each instrument!)')

        if isinstance(degree, int):
            degree = self.instruments.size * [degree]
        else:
            assert len(degree) == self.instruments.size, \
                'Provide trend degree for each instrument or one integer'

        if slopes is not None:
            assert len(slopes) == self.instruments.size, \
                'Provide trend slopes for each instrument'

        # longest_name
        ln = self._longest_name

        if plot:
            ax, _ = self.plot()
            colors = [l.get_color() for l in ax.get_lines()]
            colors = [change_color(c, 1.2) for c in colors]

        for i, instrument in enumerate(self.instruments):
            mask = self.obs == i + 1
            m = mask & self.mask

            t, v, e = self.time, self.vrad, self.svrad

            fitp = np.polyfit(t[m], v[m], degree[i], w=1 / e[m])
            if slopes and slopes[i]:
                fitp = np.polyfit(t[m], v[m] - slopes[i] * t[m], 0, w=1 / e[m])
                print(fitp)
                fitp = np.r_[slopes[i], fitp]

            if self.verbose:
                print(blue | f'coefficients  {instrument:{ln}s}: {fitp}')

            if plot:
                tt = np.linspace(t[m].min(), t[m].max(), 1000)
                # max_zorder = max([l.get_zorder() for l in ax.get_lines()])
                ax.plot(tt, np.polyval(fitp, tt), color=colors[i], lw=2,
                        zorder=3)

            # v[m] = v[m] - np.polyval(fitp, t[m])
            self.vrad[mask] = self.vrad[mask] \
                              - np.polyval(fitp, self.time[mask])

            self.each[i].vrad = self.each[i].vrad \
                              - np.polyval(fitp, self.each[i].time)

        self.periodogram_calculated = False
        self._reproduce.append(
            get_function_call_str(inspect.currentframe(), top=False))

    def detrend_jointly(self, degree=1, plot=True, ESPRESSO_fiber_offset=True,
                        HARPS_fiber_offset=True):
        """
        Remove a trend of a given degree from the RVs, adjusting one offset for
        each instrument

        Arguments
        ---------
        degree: int
            Trend degree, shared between instruments
        plot: bool
            Show a plot with the trend fit
        """
        m = self.mask
        t, v, e = self.time, self.vrad, self.svrad

        obs = self.obs.copy()

        if ESPRESSO_fiber_offset and 'ESPRESSO' in self.instruments:
            E = self.each.ESPRESSO
            if E.has_before_and_after_fibers:
                mask = self.time > self.technical_intervention
                obs[E.global_mask & mask] += 0.5

        inverse_indices = np.unique(obs, return_inverse=True)[1]

        def model(t, *p):
            coeff = p[:degree+1]
            offsets = np.r_[0.0, p[degree+1:]]
            return np.polyval(coeff, t) + offsets[inverse_indices]

        p0 = np.polyfit(t[m], v[m], degree)
        p0 = np.r_[p0, [0.0]]
        fitp = optimize.curve_fit(model, t[m], v[m], p0=p0, sigma=e[m])[0]

        if self.verbose:
            print(blue | 'coefficients:', f'{fitp[:degree+1]}')
            print(blue | '     offsets:', f'{fitp[degree+1:]}')

        pars = np.insert(fitp, degree+1, 0.0)
        coeff = pars[:degree+1]

        if plot:
            ax, _ = self.plot()

            for i, ob in enumerate(np.unique(obs)):
                ti = t[obs == ob]
                tt = np.linspace(ti.min(), ti.max(), 200)
                ax.plot(tt, np.polyval(coeff, tt) + pars[degree+1+i],
                        color='k', lw=3, zorder=3)

        v = v - model(t, *fitp)
        self.vrad = v
        self.periodogram_calculated = False
        self._reproduce.append(
            get_function_call_str(inspect.currentframe(), top=False))

    def secular_acceleration(self, epoch=55500, instruments=True, plot=True,
                             verbose=False):
        """
        Remove secular acceleration from RVs

        Arguments
        ---------
        epoch : float
            The reference epoch (DACE uses 55500, 31/10/2010)
        instruments : bool or collection of str, default True
            Only remove secular acceleration for some instruments, or for all
            if `instruments=True`.
        plot : bool, default True
            Show a plot of the RVs with the secular acceleration
        """
        if self._removed_secular_acceleration:  # don't do it twice
            return

        if self.verbose:
            print(blue | 'INFO: Removing secular acceleration from RVs')

        try:
            sa = secular_acceleration(self.star, verbose=self.verbose)
            if self.units == 'km/s':
                sa /= 1000
        except ValueError as e:
            if self.verbose:
                print(red | 'ERROR:', e)
            return
        except AttributeError:
            return

        if self.verbose and epoch is not None:
            print(f'  epoch: {epoch}')

        if plot:
            ax, _ = self.plot()
            colors = [l.get_color() for l in ax.get_lines()]
            colors = [change_color(c, 1.2) for c in colors]

        ## select only some instruments to remove SA
        all_instruments = self.instruments
        if instruments is True:
            instruments = all_instruments
        else:
            instruments_to_do = []
            for i_do in instruments:
                for inst in all_instruments:
                    if i_do in inst:
                        instruments_to_do.append(inst)
            instruments = instruments_to_do


        for i, instrument in enumerate(instruments):

            if 'HIRES' in instrument:
                # never remove it from HIRES...
                if self.verbose:
                    print(
                        yel |
                        '--> not removing secular acceleration from HIRES RVs')
                continue

            mask = self.obs == i + 1
            m = mask & self.mask
            t, v, e = self.time, self.vrad, self.svrad

            if epoch is None:  # probably not a good idea...
                epoch = t[m].mean()

            if plot:
                tt = np.linspace(t[m].min(), t[m].max(), 10)
                # max_zorder = max([l.get_zorder() for l in ax.get_lines()])
                ax.plot(tt,
                        sa * (tt - epoch) / 365.25 + v[m].mean(),
                        color=colors[i], lw=2, zorder=3)

            # # v[m] = v[m] - np.polyval(fitp, t[m])
            self.vrad[mask] = self.vrad[mask] - sa * (self.time[mask] - epoch) / 365.25

            self.each[i].vrad = self.each[i].vrad - sa * (self.each[i].time - epoch) / 365.25

        self._removed_secular_acceleration = True
        self.periodogram_calculated = False
        self._reproduce.append(
            get_function_call_str(inspect.currentframe(), top=False))

    def search_for_TESS(self, tid=None, toi=None):
        if self.verbose:
            print(blue | 'INFO: ', end='')
            print('Searching for TESS data with lightkurve.', flush=True)
            # print('Can take a while...', flush=True)

        try:
            if tid is not None:
                self._tess = TESS(TIC=tid)
                return
            elif toi is not None:
                self._tess = TESS(TOI=toi)
                return
            elif 'TOI' in self.star:
                toi = int(''.join([s for s in self.star if s.isdigit()]))
                self._tess = TESS(TOI=toi)
                return
            else:
                from astroquery.mast import Catalogs
                if self.verbose:
                    print(blue | 'INFO: ', end='')
                    print(
                        f'Querying MAST for {self.star}. Can take a while...',
                        flush=True)

                q = Catalogs.query_object(self.star, catalog="Tic",
                                          radius=0.001)
                dwarfs = q['lumclass'] == 'DWARF'

                if len(q[dwarfs]) > 0:
                    tid = q[dwarfs][0]['ID']
                    if self.verbose:
                        print(yel | 'Result:', f'found TIC ID {tid}')
                    self._tess = TESS(TIC=tid)
                    return

        except ValueError:
            if self.verbose:
                print(red | 'ERROR:',
                      f'Could not find TESS data for {self.star}')

    def search_for_superWASP(self):
        try:
            sW = photometry.query_object(self.star, catalogue='superWASP',
                                         verbose=self.verbose)
            self._superWASP = sW
        except ValueError as e:
            if self.verbose:
                print(red | 'ERROR:', e)

    def search_for_ASAS_SN(self):
        try:
            a = photometry.query_object(self.star, catalogue='ASAS-SN',
                                         verbose=self.verbose)
            self._ASAS_SN = a
        except ValueError as e:
            if self.verbose:
                print(red | 'ERROR:', e)

    def search_for_ESO(self, instrument='ESPRESSO'):
        """ Search the ESO archive for reduced spectra and add the RVs """
        warnings.warn('This is an ugly hack!')
        from .query_ESOarchive import query_ESO
        query_ESO(self.star, instrument=instrument)

    def fit_step(self, instrument, plot=True):

        if not isinstance(instrument, str):
            raise ValueError('instrument should be str')

        if self.verbose:
            print(f'Removing step-function trend from RVs of {instrument}')

        i = np.where(self.instruments == instrument)[0][0]
        mask = self.obs == i + 1
        m = mask & self.mask

        t, v, e = self.time, self.vrad, self.svrad

        def f(x, a, b, c):
            return a * np.sign(x - b) + c  # Heaviside fitting function

        p0 = [v[m].ptp() / 2, t[m][np.ediff1d(v[m]).argmax()], v[m].ptp() / 2]
        popt, _ = optimize.curve_fit(f, t[m], v[m], sigma=e[m], p0=p0,
                                     method='trf')

        if plot:
            tt = np.linspace(t[m][0], t[m][-1], 1000)
            ax, _ = self.plot()

            colors = [l.get_color() for l in ax.get_lines()]
            colors = [change_color(c, 1.2) for c in colors]
            color = colors[i]

            ax.plot(tt, f(tt, *popt), color=color, lw=2)

        self.vrad[mask] = self.vrad[mask] - f(self.time[mask], *popt)
        self._reproduce.append(
            get_function_call_str(inspect.currentframe(), top=False))

    def add_sine(self, period=None, period_range=None, adjust=False, plot=True,
                 over=1):
        """ Add a sinusoid to the RVs, optionally adjust the period """
        # try:
        #     model = self.GLS['model']
        # except AttributeError:
        #     self.gls(plot=True)
        #     model = self.GLS['model']
        self.gls(recompute=True, gatspy=True)
        model = self.GLS['model']

        if period is None:
            if period_range is None:
                p = self.GLS['period']
                pr = (max(1, p.min()), min(self.time[self.mask].ptp(),
                                           p.max()))
                model.optimizer.period_range = pr
            else:
                model.optimizer.period_range = period_range
            print(model.optimizer.period_range)
            model.fit_period = True

            period = model.find_best_periods(n_periods=1)

        if adjust:
            print(red | 'adjust is not implemented yet')
        #     m = self.mask
        #     p0 = [self.vrad[m].ptp(), period, 0., 0.]
        #     coeff = sinefit(self.time[m], self.vrad[m], self.svrad[m], p0)
        #     print(coeff)

        if self.verbose:
            print(blue | f'Adding a sinusoid at {period} days')

        if plot:
            ax, _ = self.plot()
            lin = np.linspace
            if isinstance(model, periodic.LombScargleMultiband):
                for i, instrument in enumerate(self.instruments):
                    mask = self.obs == i + 1
                    m = mask & self.mask
                    tt = lin(self.time[m].min(), self.time[m].max(), 1000)
                    sinusoid = model.predict(tt, i + 1, period)
                    ax.plot(tt, sinusoid, color='k', lw=3, zorder=3)
            else:
                ts = self.time.ptp()
                mi, ma = self.time.min(), self.time.max()
                tt = lin(mi - over * ts, ma + over * ts, 1000)
                # max_zorder = max([l.get_zorder() for l in ax.get_lines()])
                sinusoid = model.predict(tt, period)
                ax.plot(tt, sinusoid, color='k', lw=2, zorder=-1, alpha=0.5)

        if isinstance(model, periodic.LombScargleMultiband):
            sinusoid = model.predict(self.time, self.obs, period)
        else:
            sinusoid = model.predict(self.time, period)

        self.vrad = self.vrad - sinusoid
        self.periodogram_calculated = False

    def correlate(self, ax=None, axcb=None, instruments=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
        else:
            fig = plt.gcf()

        if instruments:
            #! known bug, self.mask is not considered
            which = [i for i in self.each if i.instruments[0] in instruments]
            time = np.concatenate([i.time[i.mask] for i in which])
            vrad = np.concatenate([i.vrad[i.mask] for i in which])
            fwhm = np.concatenate([i.fwhm[i.mask] for i in which])
            sc = ax.scatter(fwhm, vrad, c=time, cmap=plt.cm.viridis)
        else:
            m = self.mask
            sc = ax.scatter(self.fwhm[m], self.vrad[m], c=self.time[m],
                            cmap=plt.cm.viridis)

        # divider = make_axes_locatable(ax)
        # cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
        # fig.add_axes(cax)
        # fig.colorbar(im, cax=cax, orientation="horizontal")
        if axcb is None:
            cb = fig.colorbar(sc, orientation='horizontal')
            axcb = fig.axes[-1]
        else:
            cb = fig.colorbar(sc, cax=axcb, orientation='horizontal')

        ax.set(xlabel=f'FWHM [{self.units}]', ylabel=f'RV [{self.units}]')
        axcb.set(xlabel='Time [days]')
        return cb

    def _run_treament_steps(self, plot=True):
        if self._ran_treatment:
            return

        if not os.path.exists('individual_data_treatment.py'):
            return

        # exec(open('individual_data_treatment.py').read()) in locals()
        d = {}
        with open("individual_data_treatment.py") as f:
            code = compile(f.read(), f.name, 'exec')
            exec(code, d)
        treatment = d['treatment']

        if self.star not in treatment:  # nothing to do
            if self.verbose:
                print(blue | f'--> no data treatment steps for {self.star}')
            self._ran_treatment = True
            return

        steps, do_when_ESPRESSO_only = treatment[self.star]

        # check if there is only ESPRESSO, maybe should not do anything
        if self.instruments.size == 1 and 'ESPRESSO' in self.instruments:
            if not do_when_ESPRESSO_only:
                if self.verbose:
                    print(yel | f'-> only ESPRESSO in instruments, skipping.')
                return

        for step in steps:
            eval('self.' + step)
            if not plot: plt.close()

        self._ran_treatment = True

    def detectable_mass(self):
        self.detection_limits(plot=False, avoid=True)
        M = []
        try:
            for sim in self.simulated_systems:
                if isinstance(sim, dict):
                    d = sim
                else:
                    d = sim.d

                region = (d['trial_periods'] > self.HZ[0]) & (d['trial_periods'] < self.HZ[1])
                pfit = np.polyfit(d['trial_periods'][region], d['det_masses'][region], 1)
                yfit = np.polyval(pfit, self.HZ.mean())
                # ax.plot(self.HZ.mean(), , 's', ms=10, color='k')
                # if yfit < 10:
                #     ax.hlines(y=yfit, xmin=self.HZ.mean(), xmax=1000, color='m')
                #     ax.text(1050, yfit, str(round(yfit, 1)) + ' $M_\oplus$')
                M.append(yfit)
        except AttributeError:
            M.append(0)
        return M

    def save(self,
             prefix=None,
             file_postfix='',
             bin_text=False,
             header=True,
             planetpack=False,
             save_indicators=False,
             separate_instruments=True,
             instrument=None):

        if not separate_instruments and not isinstance(instrument, str):
            error('If not separating instruments, '
                  'must provide `instrument` name')
            return

        headertext = 'jdb\tvrad\tsvrad'

        is_all_nan = lambda x: np.isnan(x).all()

        if save_indicators:
            ind = ['fwhm', 'contrast', 'rhk', 'sindex']
            ind = ind + list(self.activity_indicators)
            ind = [i for i in ind if hasattr(self, i)]
            for i in ind:
                try:
                    v = getattr(self, i)
                    if not is_all_nan(v):
                        headertext += f'\t{i}'

                    try:
                        ev = getattr(self, 'e' + i)
                        if not is_all_nan(ev):
                            headertext += f'\t{i}_err'
                    except AttributeError:
                        pass
                    try:
                        ev = getattr(self, i + '_err')
                        if not is_all_nan(ev):
                            headertext += f'\t{i}_err'
                    except AttributeError:
                        pass
                except AttributeError:
                    pass
                    # ind.remove(i)

        fmt = ['%.6f'] + ['%.9f'] * (len(headertext.split('\t')) - 1)

        headerlines = ['-'*l for l in map(len, headertext.split('\t'))]
        headertext += '\n'
        headertext += '\t'.join(headerlines)

        if prefix is None:
            prefix = ''

        star = ''.join(self.star.split(' '))
        each = (self,) if self._child else self.each

        data = []
        filenames = []

        if separate_instruments:
            for i, individual in enumerate(each):
                if planetpack:
                    inst = individual.instruments[0]
                    filename = os.path.join(prefix, f'{star}-{inst}.rv')
                else:
                    b = ''# "_bin" if individual.is_binned else ""
                    inst = individual.instruments[0]
                    post = file_postfix
                    filename = os.path.join(prefix, f'{star}_{inst}{b}{post}.rdb')

                m = individual.mask & self.mask[self.obs == i + 1]

                if m.sum() == 0:
                    continue

                if self.verbose:
                    print(blue | '--> saving to:', filename)

                filenames.append(filename)

                d = np.c_[
                    individual.time[m],
                    individual.vrad[m],
                    individual.svrad[m]
                ]

                if save_indicators:
                    for i in ind:
                        try:
                            v, ev = getattr(individual, i)[m], getattr(individual, 'e'+i)[m]
                        except AttributeError:
                            try:
                                v, ev = getattr(individual, i)[m], getattr(individual, i+'_err')[m]
                            except AttributeError:
                                v, ev = getattr(individual, i)[m], None

                        if is_all_nan(v):
                            continue
                        elif is_all_nan(ev) or ev is None:
                            d = np.c_[d, v]
                        else:
                            d = np.c_[d, v, ev]

                data.append(d)

        else:
            inst = instrument
            if planetpack:
                filename = os.path.join(prefix, f'{star}-{inst}.rv')
            else:
                b = "_bin" if self.is_binned else ""
                post = file_postfix
                filename = os.path.join(prefix, f'{star}_{inst}{b}{post}.rdb')

            if self.verbose:
                print(blue | '--> saving to:', filename)

            filenames.append(filename)

            m = self.mask
            d = np.c_[self.time[m], self.vrad[m], self.svrad[m]]

            if save_indicators:
                for i in ind:
                    try:
                        d = np.c_[d, getattr(self, i)[m], getattr(self, 'e'+i)[m]]
                    except AttributeError:
                        try:
                            d = np.c_[d, getattr(self, i)[m], getattr(self, i+'_err')[m]]
                        except AttributeError:
                            d = np.c_[d, getattr(self, i)[m]]

            data.append(d)

        header = headertext if header else ''

        for d, filename in zip(data, filenames):
            dir_ = os.path.dirname(filename)
            if dir_ != '' and not os.path.exists(dir_):
                os.makedirs(dir_)

            np.savetxt(filename, d, header=header, comments='', delimiter='\t',
                        fmt=fmt)

        return filenames

    from ._report import report

    def _kima(self, directory=None, npmax=1, run=False, ESPRESSO_only=True,
              ncores=4, to=60, force=False, create_dir=False, edit=False,
              ESPRESSO_fiber_offset=True):
        from pykima import make_template
        star = ''.join(self.star.split())

        if directory is None:
            d = star + '_kima_analysis'
        else:
            d = directory

        # save directory
        self._kima_directory = d

        if create_dir:
            if not os.path.exists(d):
                if self.verbose: print(yelb | 'Created', d)
                os.mkdir(d)

            make_template.main(d)

        # first, check if we need to run any data treatment
        self.secular_acceleration(plot=False)

        if not self._ran_treatment:
            self._run_treament_steps()
        # and pre-calculate the hash
        self.hash

        ## save data to directory
        if create_dir:
            self.save(prefix=d, ESPRESSO_fiber_offset=ESPRESSO_fiber_offset)

        if edit:
            ## change kima_setup.cpp
            file = os.path.join(d, 'kima_setup.cpp')
            os.system(f'nano {file}')

            # ks = star + '_kima_setup.cpp'
            # if os.path.exists(ks):
            #     print('reading', ks)
            #     ks = open(ks).read()
            # else:
            #     units = '"ms"' if self.units == 'm/s' else '"kms"'
            #     if ESPRESSO_only:
            #         print(yel | 'using template', '"template_kima_setup.cpp"')
            #         assert 'ESPRESSO' in self.instruments
            #         df = self.save(prefix=d)
            #         df = [os.path.basename(f) for f in df if 'ESPRESSO' in f][0]
            #         file = 'template_kima_setup.cpp'
            #         ks = open(file).read() % (npmax, df, units)
            #     else:
            #         print(yel | 'using template',
            #             '"template_kima_setup_multi.cpp"')
            #         df = self.save(prefix=d)
            #         df = list(map(os.path.basename, df))
            #         df = ', '.join([f.center(len(f) + 2, '"') for f in df])
            #         file = 'template_kima_setup_multi.cpp'
            #         ks = open(file).read() % (npmax, df, units)

            # with open(os.path.join(d, 'kima_setup.cpp'), 'w') as f:
            #     f.write(ks)

            ## change OPTIONS
            file = os.path.join(d, 'OPTIONS')
            os.system(f'nano {file}')

            # op = star + '_OPTIONS'
            # if os.path.exists(op):
            #     if self.verbose:
            #         print(yel | 'reading', op)
            #     op = open(op).read()
            # else:
            #     if self.verbose:
            #         print(yel | 'using template', '"template_OPTIONS"')
            #     op = open('template_OPTIONS').read()

            # with open(os.path.join(d, 'OPTIONS'), 'w') as f:
            #     f.write(op)

        if run:
            pwd = os.getcwd()
            try:
                os.chdir(d)
                try:
                    old_hash = open('hash').read()
                    if old_hash == self.hash and not force:
                        print(blue | 'Data hash matches, analysis already done!')
                        os.chdir(pwd)
                        return
                    else:
                        print(blue | 'old hash:', old_hash)
                        print(blue | 'new hash:', self.hash)
                except FileNotFoundError:
                    pass

                if to == -1:
                    cmd = 'kima-run -t %d -b -q &' % ncores
                else:
                    cmd = 'kima-run -t %d --timeout %d -b -q &' % (ncores, to)

                print(green | 'starting kima...', end=' ', flush=True)
                # os.system(cmd)
                subprocess.check_call(cmd.split()[:-1])
                print(green | 'finished!')
                open('hash', 'w').write(self.hash)

                # store the last kima analysis
                time_stamp = datetime.now(timezone.utc).isoformat().split('.')[0]
                open(f'{star}.lastkima.dat', 'w').write(time_stamp)
                self._lastkima = time_stamp

            except Exception as e:
                raise e
            finally:
                os.chdir(pwd)

    def kima(self, *args, **kwargs):
        self._kima(*args, **kwargs)

    def juliet(self, np=1, sectors='all'):
        from .juliet_fit import fit
        tess = self.tess
        self._juliet = fit(tess, np, sectors)

    from ._actin import actin

    def gp_both(self, nplanets=1):
        import scipy.optimize as op
        from pykima.keplerian import keplerian
        from george import GP
        from george.kernels import ExpSine2Kernel as PER, ExpSquaredKernel as SE
        from george.modeling import Model

        QP = lambda η1, η2, η3, η4: η1**2 * SE(η2**2) * PER(2/η4**2, np.log(η3))
        obs = self.obs - 1
        offset_times = self.offset_times

        class SimpleMean(Model):
            parameter_names = ('C', 'of')
            def get_value(self, t):
                if t.size == obs.size:
                    return np.array([self.C, self.of])[obs]
                else:
                    C = np.zeros_like(t)
                    C[t > offset_times] = self.of
                    return C

        class Keplerian(Model):
            parameter_names = ('P', 'K', 'e', 'φ', 'Tp', 'C', 'of')
            def get_value(self, t):
                P, K, e, φ, Tp = self.P, self.K, self.e, self.φ, self.Tp
                if t.size == obs.size:
                    C = np.array([self.C, self.of])[obs]
                    return C + keplerian(t, P, K, e, φ, Tp, 0.0)
                else:
                    C = np.zeros_like(t)
                    C[t > offset_times] = self.of
                    return C + keplerian(t, P, K, e, φ, Tp, self.C)

        class TwoKeplerian(Model):
            parameter_names = (
                'P1', 'K1', 'e1', 'φ1', 'Tp1',
                'P2', 'K2', 'e2', 'φ2', 'Tp2',
                'C', 'of')
            def get_value(self, t):
                P, K, e, φ, Tp = \
                    [self.P1, self.P2], [self.K1, self.K2], [self.e1, self.e2],\
                    [self.φ1, self.φ2], [self.Tp1, self.Tp2]

                if t.size == obs.size:
                    C = np.array([self.C, self.of])[obs]
                    return C + keplerian(t, P, K, e, φ, Tp, 0.0)
                else:
                    C = np.zeros_like(t)
                    C[t > offset_times] = self.of
                    return C + keplerian(t, P, K, e, φ, Tp, self.C)

        if nplanets == 1:
            mean = Keplerian(11.2, 1.5, 0.2, 0, 58531, 0, 0)
        elif nplanets == 2:
            mean = TwoKeplerian(11.2, 1.5, 0.2, 0, 58531,
                                5.1, 0.5, 0, 0, 58531,
                                0, 0)

        gp1 = GP(kernel=QP(1, 10, 80, 0.5), mean=mean)
        gp1.compute(self.time[self.mask], self.svrad[self.mask])
        # print(gp1.log_likelihood(self.vrad[self.mask]))

        gp2 = GP(kernel=QP(10, 10, 80, 0.5), mean=SimpleMean(0.0, 0.0))
        gp2.compute(self.time[self.mask], self.efwhm[self.mask])

        shared = [
            'kernel:k1:k2:metric:log_M_0_0',
            'kernel:k2:gamma',
            'kernel:k2:log_period'
        ]

        def nll(vector, *y):
            # print(vector)
            for par, val in zip(gp1.get_parameter_dict().keys(), vector):
                gp1.set_parameter(par, val)
                if par in shared:
                    gp2.set_parameter(par, val)
            nll1 = gp1.nll(gp1.get_parameter_vector(), y[0])
            nll2 = gp2.nll(gp2.get_parameter_vector(), y[1])
            return nll1 + nll2

        # return gp1, gp2, nll
        print(nll(gp1.get_parameter_vector(), self.vrad, self.fwhm))

        p0 = gp1.get_parameter_vector()
        bounds = [
            # (1, self.time.ptp()),
            (11, 11.5), # P1
            (0.5, 10), # K1
            (0, 1), # e1
            (None, None), # φ1
            (None, None), # Tp1
        ]

        if nplanets == 2:
            bounds.append((5, 5.5))
            bounds.append((0.1, 1))
            bounds.append((0, 0.1))

        bounds = bounds + (gp1.vector_size - len(bounds))*[(None, None)]
        # bounds[-1] = (np.log(70), np.log(100))
        print(bounds)

        results = op.minimize(nll, p0, method="L-BFGS-B", #jac=gp.grad_nll,
                              args=(self.vrad, self.fwhm), bounds=bounds)

        # Update the kernel and print the final log-likelihood.
        gp1.set_parameter_vector(results.x)
        print(nll(gp1.get_parameter_vector(), self.vrad, self.fwhm))
        return gp1, gp2, results

