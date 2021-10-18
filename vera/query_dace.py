""" This script queries the DACE platform for data of a particular star """

import sys
import os
from glob import glob
import tarfile
import argparse
import requests
from collections import abc
from argparse import RawTextHelpFormatter

import numpy as np
from tqdm import tqdm
from dace.spectroscopy import Spectroscopy

from .utils import error, info
from .globals import _ramp_up, COMMISSIONING


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Query DACE for RV observations of a given star. \n"
        "ATTENTION: this script joins RVs from all modes (HR11, HR21, etc)!\n\n"
        "Before you use:\n"
        "  - install the DACE Python client API\n"
        "  - create a ~/.dacerc file with your API key\n"
        "    (see https://dace.unige.ch/tutorials/?tutorialId=10)")

    parser.add_argument("star", help="The star name", type=str)
    parser.add_argument("--inst", type=str,
                        help="Select a specific instrument")
    parser.add_argument("--rdb", action="store_true",
                        help="Save the observations in individual rdb files")
    parser.add_argument("--ms", action="store_true",
                        help="Save the observations in m/s; default is km/s")
    parser.add_argument("--save-versions", action="store_true",
                        help="Save pipeline versions in different rdb files")
    parser.add_argument("--keep-mode", type=str, default='HR11',
                        help="Which instrument mode to keep")
    parser.add_argument("--no-commissioning", action="store_true",
                        help="Remove ESPRESSO points from the commissioning")
    parser.add_argument("--write-dates", action="store_true",
                        help="Write the dates of the observations")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print more information")

    args = parser.parse_args()
    return args


def case_insensitive_glob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob(''.join(map(either, pattern)))


def dive_in_dict(d):
    k = d.keys()
    while len(k) == 1:
        d = d[list(k)[0]]
        k = d.keys()
    return d


def dive_in_dict_2(d):
    k = d.keys()
    while 'rv' not in k:
        d = d[list(k)[0]]
        k = d.keys()
    return d


def length_observations(d, instruments):
    N = 0
    for i in instruments:
        for p in list(d[i].keys()):
            for m in d[i][p].keys():
                N += len(d[i][p][m]['texp'])
            break  # only 1 pipe
    return N


def n_pipeline_versions(d):
    return [len(d[inst].keys()) for inst in d.keys()]


def rjd_to_mjd(rjd):
    """
    RJD (Reduced Julian Date)
        days elapsed since 1858-11-16T12Z (JD 2400000.0)
    MJD (Modified Julian Date)
        days elapsed since 1858-11-17T00Z (JD 2400000.5)
    This function transforms RJD in MJD
    """
    return rjd - 0.5


def get_pipe_dict(d, i):
    """
    given a dictionary d for a given instrument, return the dictionary for the
    ith pipeline version
    """
    pipe_versions = list(d.keys())
    k = pipe_versions[i]
    mode = list(d[k].keys())[0]
    return d[k][mode]


def dict_tree(d, level=0, upto=1):
    for k in sorted(d.keys()):
        print(level * '  ' + '-', k)
        if isinstance(d[k], dict) and level + 1 < upto:
            dict_tree(d[k], level=level + 1, upto=upto)


def decode_dict(d):
    instruments = sorted(list(d.keys()))
    pipemode = []
    for inst in instruments:
        this_pipemode = []
        # pipemode[inst] = []
        for pp in sorted(d[inst].keys(), reverse=True):
            if '-' in pp:
                pipe, mode = pp.split('-')
                mode_name = list(d[inst][pp].keys())[0]
            else:
                pipe = pp
                mode = sorted(list(d[inst][pipe].keys()))[0]
                mode_name = mode
            this_pipemode.append((pipe, mode, mode_name))
        pipemode.append(this_pipemode)
    return instruments, pipemode


def nested_dict_iter(nested, key=None):
    """ Return an iterator through all the nested key,value pairs of nested """
    for ikey, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from nested_dict_iter(value, key)
        else:
            if key is None:
                yield ikey, value
            else:
                if ikey == key:
                    yield ikey, value


def get_arrays(d):
    """ return the arrays from dictionary d """
    get = d.__getitem__

    t, r, e = np.array(list(map(get, ('rjd', 'rv', 'rv_err'))))
    #! 22/11/2019 João Faria
    #! DACE output has the key named "rjd", but the values are actually in BJD
    #! so we don't actually need to do the following
    # t = rjd_to_mjd(t)

    # sort by time
    ind = np.argsort(t)
    t, r, e = t[ind], r[ind], e[ind]
    # m/s -> km/s
    r /= 1000
    e /= 1000

    fwhm, efwhm = np.array(list(map(get, ('fwhm', 'fwhm_err'))))
    fwhm = fwhm.astype(float)
    efwhm = efwhm.astype(float)
    # m/s -> km/s
    fwhm /= 1000
    efwhm /= 1000

    ctst, ectst = np.array(list(map(get, ('contrast', 'contrast_err'))))
    ctst = ctst.astype(float)
    ectst = ectst.astype(float)
    # units are %

    rhk, erhk = np.array(list(map(get, ('rhk', 'rhk_err'))))
    rhk = rhk.astype(float)
    erhk = erhk.astype(float)

    m = (rhk == -99999)
    rhk[m] = np.nan
    erhk[m] = np.nan

    sindex, esindex = np.array(list(map(get, ('sindex', 'sindex_err'))))
    sindex = sindex.astype(float)
    esindex = esindex.astype(float)

    # naindex, enaindex = np.array(
    #     list(map(get, ('naindex', 'naindex_err'))))
    # naindex = naindex.astype(float)
    # enaindex = enaindex.astype(float)

    drs_qc = np.array(d['drs_qc'])

    return (
        t,
        r,
        e,
        fwhm,
        efwhm,
        ctst,
        ectst,
        rhk,
        erhk,
        sindex,
        esindex,
        # naindex, enaindex,
        drs_qc)


def separate_19_21(rv_data):
    _19 = 'ESPRESSO19'
    _21 = 'ESPRESSO21'
    if _19 in rv_data:  # actually do something
        rv_data[_21] = {}
        for pipe in rv_data[_19]:
            rv_data[_21][pipe] = {}
            for mode in rv_data[_19][pipe]:
                rv_data[_21][pipe][mode] = {}
                mask = np.array(rv_data[_19][pipe][mode]['rjd']) > _ramp_up
                for key, val in rv_data[_19][pipe][mode].items():
                    rv_data[_19][pipe][mode][key] = list(np.array(val)[~mask])
                    rv_data[_21][pipe][mode][key] = list(np.array(val)[mask])
                # no observations left on ESPRESSO19
                if np.sum(~mask) == 0:
                    rv_data.pop(_19)

    return rv_data


translate = {
    'Barnard': 'HIP87937',
    'GJ541': 'GJ54.1',
    'PiMen': 'Pi Men',
    'MASCARA1': 'MASCARA-1',
}


def get_observations(star, instrument=None, rdb=False, ms=False,
                     save_versions=False, keep_mode='HR11',
                     remove_ESPRESSO_commissioning=False, write_dates=False,
                     verbose=True, raise_exit=False):
    """ Get radial velocity observations for `star` from DACE.

    Parameters
    ----------
    star : str
        The name of the star
    instrument : str
        Get only observations from this instrument. Only used if `rdb` is True
    rdb : bool, default False
        Whether to save the observations in .rdb files, for each instrument
        separately. The filename will also contain a pipeline version id.
    ms : bool, default False
        Whether to save .rdb files in m/s.
    save_versions : bool
        If True, save different .rdb files for each version of the instrument's
        pipeline. Only used if `rdb` is True.
    keep_mode : str, optional, default 'HR11'
        Which ESPRESSO mode to keep, if multiple are present
    remove_ESPRESSO_commissioning : bool
        Remove observations from the ESPRESSO commissioning.
    write_dates : bool
        Write the dates of the observations and exit.
    verbose : bool
        Print stuff while running.
    """
    if verbose:
        info('Starting the DACE Python API')
    if star in translate:
        if verbose:
            print(f'{star} -> {translate[star]}')
        star = translate[star]

    # Gives you a dict with instruments as keys and list of data as values
    try:
        # rv_data = obs.get_rv_by_instrument(star)
        # rv_data = Dace.retrieve_spectro_timeseries(star)
        rv_data = Spectroscopy.get_timeseries(star)
    except requests.RequestException:
        E = requests.RequestException
        raise E(f'Cannot find "{star}" in DACE') from None
    except KeyError:
        print(f'Cannot find "{star}" in DACE')
        if raise_exit:
            sys.exit(1)
        raise ValueError(f'Cannot find "{star}" in DACE') from None

    # rv_data is organized as
    #   key - instrument
    #       key - pipeline version
    #           key - mode (?)
    #               keys - rjd, rv, rv_err, etc

    # before anything else, separate ESPRESSO19 and ESPRESSO21
    rv_data = separate_19_21(rv_data)

    instruments = rv_data.keys()
    # number of pipeline versions per instrument
    npipes = n_pipeline_versions(rv_data)

    if instrument and not np.any([instrument in k for k in rv_data.keys()]):
        raise ValueError(f'No observations of instrument {instrument}')

    # it seems pipeline versions are always sorted newer -> older, so we
    # will assume for now that this will always be the case
    #! 22/11/2019 this DOES NOT seem to be the case anymore
    #! 29/11/2019 we need to sort the pipeline versions!
    # need to reverse sort rv_data[inst] by its keys
    for inst in instruments:
        rv_data[inst] = dict(sorted(rv_data[inst].items(), reverse=True))

    if verbose:
        info('RVs available from', show_info=False)
        for inst in instruments:
            print(8 * ' ', inst)
            pipelines = list(rv_data[inst].keys())
            for i, pipe in enumerate(pipelines):
                mode = list(rv_data[inst][pipe].keys())[0]
                N = len(rv_data[inst][pipe][mode]['rjd'])
                print(10 * ' ', f'{pipe} ({N} observations)')
        # for k, npipe in zip(instruments, npipes):
        #     N = len(dive_in_dict_2(rv_data[k])['rjd'])
        #     print('   ', k,
        #           f'({N})', f'[{npipe} versions]' if npipe > 1 else '')

    if write_dates:
        for inst in instruments:
            # skip if selecting a specific instrument and inst doesn't match it
            if instrument is not None and inst != instrument:
                continue
        N = len(dive_in_dict_2(rv_data[instrument])['rjd'])
        d = dive_in_dict_2(rv_data[instrument])
        for f in d['raw_file']:
            print(f)

    if rdb:
        if ms:
            factor = 1e3
            msg = 'RVs will be in m/s!'
        else:
            factor = 1
            msg = 'RVs will be in km/s!'

        if verbose:
            info('Saving the observations in .rdb files. ' + msg)

        files = []
        for inst in instruments:
            # skip if selecting a specific instrument and inst doesn't match it
            if instrument is not None:
                match = inst == instrument or instrument in inst
                if not match:
                    continue

            pipelines = list(rv_data[inst].keys())

            if not save_versions:
                # After reverse sorting above, the first pipeline is always the
                # newest. But we still need to figure out which mode to keep.
                if any([keep_mode in pipe for pipe in pipelines]):
                    pipelines = [
                        pipe for pipe in pipelines if keep_mode in pipe
                    ][:1]
                else:
                    pipelines = pipelines[:1]

            for pipe in pipelines:
                filename = f"{''.join(star.split())}_{inst}_{pipe}.rdb"
                files.append(filename)
                if verbose:
                    print('   ', filename)

                d = dive_in_dict_2(rv_data[inst][pipe])

                t, r, e, \
                    fwhm, efwhm, \
                    ctst, ectst, \
                    rhk, erhk, \
                    sindex, esindex, \
                    drs_qc = get_arrays(d)

                if inst == 'ESPRESSO' and remove_ESPRESSO_commissioning:
                    if verbose:
                        print('Removing commissioning data')

                    ind = t > COMMISSIONING
                    t = t[ind]
                    r = r[ind]
                    e = e[ind]
                    fwhm = fwhm[ind]
                    efwhm = efwhm[ind]
                    ctst = ctst[ind]
                    ectst = ectst[ind]
                    rhk = rhk[ind]
                    erhk = erhk[ind]
                    sindex = sindex[ind]
                    esindex = esindex[ind]
                    drs_qc = drs_qc[ind]

                # header = 'jdb\tvrad\tsvrad\n---\t----\t-----'
                # header = 'jdb\tvrad\tsvrad\tfwhm\tefwhm\tcontrast\tecontrast\n'
                # header =
                # 'jdb\tvrad\tsvrad\tfwhm\tefwhm\tcontrast\tecontrast\trhk\terhk\n'
                header = '\t'.join([
                    'jdb', 'vrad', 'svrad', 'fwhm', 'efwhm', 'contrast',
                    'econtrast', 'rhk', 'erhk', 'sindex', 'esindex', 'drs_qc'
                ]) + '\n'

                header += '\t'.join(
                    ['-' * len(s) for s in header.strip().split('\t')])

                fmt = ['%.6f'] + 10 * ['%12.9f'] + ['%d']

                np.savetxt(
                    filename, np.c_[t, r * factor, e * factor, fwhm * factor,
                                    efwhm * factor, ctst, ectst, rhk, erhk,
                                    sindex, esindex, drs_qc], header=header,
                    comments='', delimiter='\t', fmt=fmt)

        rv_data['files'] = files

    return star, rv_data


def get_CCFs(star, instrument='ESPRESSO', directory=None, verbose=True,
             limit=None, check_existing=True, ask=True):
    """
    Try to download S1D files from DACE.

    Arguments
    ---------
    star : str
        The name of the star
    instrument : str or iterable of str
        The instrument(s) for which to look for data
    verbose : bool (optional, default True)
        Print stuff while running.
    limit : int (optional)
        Only download the first `limit` files
    check_existing : bool (optional, default True)
        Check if files already exist locally before downloading
    ask : bool (optional, default True)
        Ask for user confirmation before downloading
    """

    if star in translate:
        if verbose:
            print(f'{star} -> {translate[star]}')
        star = translate[star]

    if directory is None:
        todir = f'{star}_DACE_downloads'
    else:
        todir = directory

    # Gives you a dict with instruments as keys and list of data as values
    try:
        rv_data = Spectroscopy.get_timeseries(star)
    except requests.RequestException:
        E = requests.RequestException
        raise E(f'Cannot find "{star}" in DACE') from None

    # #! new organization of data by DACE team (without warning! argh!!)
    # if instrument == 'ESPRESSO':
    #     instrument = ('ESPRESSO18', 'ESPRESSO19')

    # before anything else, separate ESPRESSO19 and ESPRESSO21
    rv_data = separate_19_21(rv_data)

    if isinstance(instrument, str):
        instruments = (instrument, )
    else:
        instruments = instrument

    missing_files = []
    for instrument in instruments:
        #! pipeline versions are not sorted
        # reverse sort rv_data[inst] by its keys -> latest pipeline is the first
        rv_data[instrument] = dict(
            sorted(rv_data[instrument].items(), reverse=True))

        info('Checking data for ' + instrument)
        try:
            d = get_pipe_dict(rv_data[instrument], 0)
        except IndexError:
            error(f'  No observations found for {instrument}')
            continue

        N = limit or None
        raw_files = d['raw_file'][:N]

        raw_files = [os.path.basename(f) for f in raw_files]
        raw_files = [f.replace('.fits', '') for f in raw_files]
        if check_existing:
            which_exist = []
            for f in raw_files:
                pattern = os.path.join(todir, f) + '*ccf*'
                if len(case_insensitive_glob(pattern)) > 0:
                    which_exist.append(True)
                else:
                    which_exist.append(False)

            if all(which_exist):
                info(f'   All files already exist in "{todir}"',
                     show_info=False)
                continue

            raw_files = np.array(raw_files)[np.logical_not(which_exist)]

            if len(raw_files) > 0:
                info('   Missing some files', show_info=False)

            for f in raw_files:
                missing_files.append(f)

    N = len(missing_files)
    if N == 0:
        return todir

    if ask:
        info(f'Will attempt to download {N} files (to {todir})')
        info('Are you sure? (Y/n) ', show_info=False, end='')
        answer = input().lower()
        if answer == 'n':
            info('Doing nothing')
            return

    if not os.path.isdir(todir):
        os.mkdir(todir)
    file = 'result.tar.gz'

    if verbose:
        info('Downloading...')

    try:
        Spectroscopy.download_files(
            'ccf', missing_files[:limit],
            output_full_file_path=os.path.join(todir, file))
    except TypeError:
        Spectroscopy.download_files('ccf', missing_files[:limit],
                                    output_directory=todir,
                                    output_filename=file)

    if verbose:
        info('Extracting .fits files')

    output = os.path.join(todir, file)
    tar = tarfile.open(output, "r:gz")
    for member in tar.getmembers():
        if member.isreg():  # skip if the TarInfo is not a file
            member.name = os.path.basename(member.name)  # remove the path
            tar.extract(member, todir)
    os.remove(output)

    return todir


def get_spectra(star, instrument='ESPRESSO', directory=None, verbose=True,
                limit=None, only_mode=None, check_existing=True, ask=True):
    """
    Try to download S1D files from DACE.

    Arguments
    ---------
    star : str
        The name of the star
    instrument : str or iterable of str
        The instrument(s) for which to look for data
    verbose : bool (optional, default True)
        Print stuff while running.
    limit : int (optional)
        Only download the first `limit` files (per instrument)
    only_mode : str (optional)
        If provided, select only E2DS files matching this mode
    check_existing : bool (optional, default True)
        Check if files already exist locally before downloading
    ask : bool (optional, default True)
        Ask for user confirmation before downloading
    """

    if star in translate:
        if verbose:
            print(f'{star} -> {translate[star]}')
        star = translate[star]

    if directory is None:
        todir = f'{star}_DACE_downloads'
    else:
        todir = directory

    # Gives you a dict with instruments as keys and list of data as values
    try:
        rv_data = Spectroscopy.get_timeseries(star)
    except requests.RequestException:
        E = requests.RequestException
        raise E(f'Cannot find "{star}" in DACE') from None

    # #! new organization of data by DACE team (without warning! argh!!)
    # if instrument == 'ESPRESSO':
    #     instrument = ('ESPRESSO18', 'ESPRESSO19')

    # before anything else, separate ESPRESSO19 and ESPRESSO21
    rv_data = separate_19_21(rv_data)

    if isinstance(instrument, str):
        instruments = (instrument, )
    else:
        instruments = instrument

    all_instruments, pipemodes = decode_dict(rv_data)
    missing_files = []

    for i, pipemode in zip(all_instruments, pipemodes):
        if i not in instruments:
            continue

        instrument = i

        #! In this function and by default, we're interested in the latest
        #! pipeline, all modes. But if `only_mode` is given, select only that.
        latest_pipeline, *_ = pipemode[0]

        print('Checking data for', instrument)

        raw_files = []
        for pipe, mode, mode_name in pipemode:
            if only_mode is not None and only_mode not in (mode, mode_name):
                continue

            if pipe == latest_pipeline:
                if pipe in rv_data[instrument]:
                    pm = pipe
                else:
                    pm = '-'.join([pipe, mode])

                files = rv_data[instrument][pm][mode_name]['raw_file']
                for f in files:
                    raw_files.append(f)

        N = limit or None
        raw_files = raw_files[:N]

        raw_files = [os.path.basename(f) for f in raw_files]
        raw_files = [f.replace('.fits', '') for f in raw_files]

        if check_existing:
            p = os.path
            endings = ('_S1D_A.fits', '_s1d_A.fits')

            def exists(f):
                return any(
                    [p.exists(p.join(todir, f) + end) for end in endings])

            which_exist = [exists(f) for f in raw_files]

            if all(which_exist):
                print(f'   All files already exist in "{todir}"')
                continue

            raw_files = np.array(raw_files)[np.logical_not(which_exist)]

            if len(raw_files) > 0:
                print(f'   Missing some ({len(raw_files)}) files')

            for f in raw_files:
                missing_files.append(f)

    N = len(missing_files)
    if N == 0:
        return todir

    if ask:
        s = 16 * N  # ~16MB for each S1D
        print(f'Will attempt to download {N} files (to {todir})')
        print(f'This is about {s} MB.  Are you sure? ', end='(Y/n) ')
        answer = input().lower()
        if answer == 'n':
            print('Doing nothing')
            return 'stopped'

    if not os.path.isdir(todir):
        os.mkdir(todir)
    file = 'result.tar.gz'

    if verbose:
        print('Downloading...', flush=True)

    try:
        Spectroscopy.download_files(
            's1d', missing_files,
            output_full_file_path=os.path.join(todir, file))
    except TypeError:
        Spectroscopy.download_files('s1d', missing_files,
                                    output_directory=todir,
                                    output_filename=file)

    if verbose:
        print('Extracting .fits files')

    output = os.path.join(todir, file)
    tar = tarfile.open(output, "r:gz")
    for member in tar.getmembers():
        if member.isreg():  # skip if the TarInfo is not a file
            member.name = os.path.basename(member.name)  # remove the path
            tar.extract(member, todir)

    return todir


def get_E2DS(star, instrument='ESPRESSO', directory=None, verbose=True,
             limit=None, only_mode=None, check_existing=True, ask=True):
    """
    Try to download E2DS files from DACE.

    Arguments
    ---------
    star : str
        The name of the star
    instrument : str or iterable of str
        The instrument(s) for which to look for data
    verbose : bool (optional, default True)
        Print stuff while running.
    limit : int (optional)
        Only download the first `limit` files
    only_mode : str (optional)
        If provided, select only E2DS files matching this mode
    check_existing : bool (optional, default True)
        Check if files already exist locally before downloading
    ask : bool (optional, default True)
        Ask for user confirmation before downloading
    """

    if star in translate:
        if verbose:
            print(f'{star} -> {translate[star]}')
        star = translate[star]

    if directory is None:
        todir = f'{star}_DACE_downloads'
    else:
        todir = directory

    # Gives you a dict with instruments as keys and list of data as values
    try:
        # rv_data = Dace.retrieve_spectro_timeseries(star)
        rv_data = Spectroscopy.get_timeseries(star)
    except requests.RequestException:
        E = requests.RequestException
        raise E(f'Cannot find "{star}" in DACE') from None

    #! new organization of data by DACE team (without warning! argh!!)
    # if instrument == 'ESPRESSO':
    #     instrument = ('ESPRESSO18', 'ESPRESSO19')

    if isinstance(instrument, str):
        instruments = (instrument, )
    else:
        instruments = instrument

    all_instruments, pipemodes = decode_dict(rv_data)
    missing_files = []

    for i, pipemode in zip(all_instruments, pipemodes):
        if i not in instruments:
            continue

        instrument = i

        #! In this function and by default, we're interested in the latest
        #! pipeline, all modes. But if `only_mode` is given, select only that.
        latest_pipeline, *_ = pipemode[0]

        print('Checking data for', instrument)

        raw_files = []
        for pipe, mode, mode_name in pipemode:
            if only_mode is not None and only_mode not in (mode, mode_name):
                continue

            if pipe == latest_pipeline:
                if pipe in rv_data[instrument]:
                    pm = pipe
                else:
                    pm = '-'.join([pipe, mode])

                files = rv_data[instrument][pm][mode_name]['raw_file']
                for f in files:
                    raw_files.append(f)

        N = limit or None
        raw_files = raw_files[:N]

        raw_files = [os.path.basename(f) for f in raw_files]
        raw_files = [f.replace('.fits', '') for f in raw_files]

        if check_existing:
            p = os.path
            endings = ('_S2D_A.fits', '_e2ds_A.fits')

            def exists(f):
                return any(
                    [p.exists(p.join(todir, f) + end) for end in endings])

            which_exist = [exists(f) for f in raw_files]

            if all(which_exist):
                print(f'   All files already exist in "{todir}"')
                continue

            raw_files = np.array(raw_files)[np.logical_not(which_exist)]

            if len(raw_files) > 0:
                print(f'   Missing some ({len(raw_files)}) files')

            from astropy.time import Time
            for f in raw_files:
                # r.ESPRE.2020-12-04T05:15:52.445
                # print(Time(f[8:]))
                missing_files.append(f)

    N = len(missing_files)
    if N == 0:
        return todir

    if ask:
        s = 63 * N * 2  # ~63MB for each S2D plus the S2D_BLAZE
        print(f'Will attempt to download {N} files (to {todir})')
        print(f'This is about {s} MB.  Are you sure? ', end='(Y/n) ')
        answer = input().lower()
        if answer == 'n':
            print('Doing nothing')
            return 'stopped'

    if not os.path.isdir(todir):
        os.mkdir(todir)
    file = 'result.tar.gz'

    if verbose:
        print('Downloading...', flush=True)

    try:
        Spectroscopy.download_files(
            'e2ds', missing_files,
            output_full_file_path=os.path.join(todir, file))
    except TypeError:
        Spectroscopy.download_files('e2ds', missing_files,
                                    output_directory=todir,
                                    output_filename=file)

    if verbose:
        print('Extracting .fits files')

    output = os.path.join(todir, file)
    tar = tarfile.open(output, "r:gz")
    for member in tqdm(tar.getmembers()):
        if member.isreg():  # skip if the TarInfo is not a file
            member.name = os.path.basename(member.name)  # remove the path
            tar.extract(member, todir)

    return todir


if __name__ == '__main__':
    args = _parse_args()
    # print(args)

    get_observations(star=args.star, instrument=args.inst, rdb=args.rdb,
                     ms=args.ms, keep_mode=args.keep_mode,
                     write_dates=args.write_dates, verbose=args.verbose,
                     raise_exit=True)
