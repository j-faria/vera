"""
The KOBE experiment: a legacy survey at Calar Alto Observatory
PI: Jorge Lillo-Box
https://kobe.caha.es/
"""

from pprint import pprint
from requests import RequestException

from .utils import red
from .DACE import DACERV, escape, deescape

KOBE_targets = [f'KOBE-{i:03d}' for i in range(1, 100)]


class KOBE():
    """
    This class holds information about KOBE targets.

    To access the RVs of a given target T use `KOBE.T`
    All symbols not allowed by Python (-, +, or .) have been replaced with an
    underscore _. For KOBE targets, this means 'KOBE-001' --> KOBE.KOBE_001

    .target : RV
        Instance of the `RV` class for a given target
    """

    # kwargs for RV with their default values
    local_first = False  # try to read local files before querying DACE
    verbose = True  # what it says
    bin_nightly = False  # bin the observations nightly
    sigmaclip = False  # sigma-clip RVs, FWHM, and other observations
    maxerror = 50  # max. RV error allows, mask points with larger errors
    adjust_means = True
    keep_pipeline_versions = True  # keep pipeline versions
    download_TESS = False  # try to download TESS data
    remove_secular_acceleration = True  # subtract secular acceleration from RVs
    mask_drs_qc0 = False  # whether to mask out observations with drs_qc=0

    def __init__(self):
        self._print_errors = True
        self._attributes = set()
        self._ignore_attrs = list(self.__dict__.keys())

    @property
    def settings(self):
        msg = "Use .set(setting=True/False) to change each setting.\n"
        msg += "For example KOBE.set(bin_nightly=False, verbose=True)\n"
        print(msg)
        pprint(self._kwargs)

    @property
    def _kwargs(self):
        k = {
            'verbose': self.verbose,
            'bin': self.bin_nightly,
            'sigmaclip': self.sigmaclip,
            'maxerror': self.maxerror,
            'adjust_means': self.adjust_means,
            'keep_pipeline_versions': self.keep_pipeline_versions,
            'remove_secular_acceleration': self.remove_secular_acceleration,
            'mask_drs_qc0': self.mask_drs_qc0,
            'tess': self.download_TESS,
        }
        return k

    def set(self, verbose=None, bin_nightly=None, adjust_means=None,
            local_first=None, sigmaclip=None, maxerror=None,
            download_TESS=None, keep_pipeline_versions=None,
            remove_secular_acceleration=None, mask_drs_qc0=None,
            reload_all=True):

        def _not_none_and_different(val, name):
            return val is not None and val != getattr(self, name)

        change = False

        if _not_none_and_different(verbose, 'verbose'):
            self.verbose = verbose
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

        if _not_none_and_different(mask_drs_qc0, 'mask_drs_qc0'):
            self.mask_drs_qc0 = mask_drs_qc0
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
                print(red | f'ERROR: {star} no local data?')
            return

    def _from_DACE(self, star):
        try:
            return DACERV.from_DACE(star, **self._kwargs)
        except (KeyError, RequestException, ValueError):
            if self._print_errors:
                print(red | f'ERROR: {star} no data found in DACE?')
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

        # hack to make KOBE._001 work
        if star.startswith('0'):
            star = f'KOBE-{star}'

        if star in KOBE_targets:
            if self.verbose:
                print(f'Found "{star}" in KOBE target list')
        else:
            print(f'Cannot find "{star}" in KOBE target list.')
            return


        if self.local_first:
            t = self._from_local(star)
            if t is None:
                t = self._from_DACE(star)
        else:
            t = self._from_DACE(star)

        if t is None:
            return

        setattr(self, escape(star), t)
        self._attributes.add(escape(star))

        return t
