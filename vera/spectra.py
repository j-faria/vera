from os.path import basename, dirname, exists
from glob import glob
from collections import Iterable

import numpy as np
import matplotlib.pyplot as plt
from cached_property import cached_property

from astropy.io import fits
from iCCF.keywords import getBJD
from tqdm.std import tqdm


broad_tellurics = {
    '$O_2$': ((6270, 6330), (6860, 6970), (7570, 7700),),
    '$H_20$': ((7140, 7450), )  # (7850, 8550),),
}


def add_telluric_regions(ax):
    ylims = np.array(ax.get_ylim())
    y0 = - ylims.ptp() / 10
    y1 = y0 + 0.02*ylims.ptp()

    for mol, regions in broad_tellurics.items():
        for region in regions:
            ax.vlines([region], ymin=y0, ymax=y1, color='r')
            ax.hlines([y0], xmin=region[0], xmax=region[1], color='r')
            ax.text(np.mean(region), y0 - 0.1 * abs(y0), mol, ha='center',
                    va='top', color='r', fontsize=8)


class Spectrum:
    def __init__(self, filename):
        self.s1d = 'S1D' in filename
        # self.e2ds = 'S2D' in filename

        # hdul = fits.open(filename)
        with fits.open(filename) as hdul:
            self.filename = filename
            s2d_exists = self._check_for_s2d()
            if s2d_exists:
                self.s2d = self._read_s2d()
            else:
                self.s2d = None

            inst = hdul[0].header['INSTRUME']
            if inst == 'ESPRESSO':
                self.wave = hdul[1].data['wavelength']
                self.wave_air = hdul[1].data['wavelength_air']
                self.flux = hdul[1].data['flux']
                self.error = hdul[1].data['error']
                self.quality = hdul[1].data['quality']
                self.mask = self.quality == 0
            elif inst == 'HARPS':
                self.wave = hdul[0].header['CRVAL1'] + np.arange(
                    0, hdul[0].header['NAXIS1'], hdul[0].header['CDELT1'])
                self.flux = hdul[0].data

    def __repr__(self):
        r = f'Spectrum({basename(self.filename)})'
        return r

    def __len__(self):
        return 1

    # a convenience method to imitate the CCF constructor
    @classmethod
    def from_file(cls, file, hdu_number=1, data_index=-1, sort_bjd=True,
                  progress=True, **kwargs):
        """
        Create a `Spectrum` object from one or more fits files.

        Parameters
        ----------
        file : str or list of str
            The name(s) of the fits file(s)
        hdu_number : int, default = 1
            The index of the HDU list which contains the spectrum
        data_index : int, default = -1
            The index of the .data array which contains the CCF. The data will
            be accessed as ccf = HDU[hdu_number].data[data_index,:]
        sort_bjd : bool
            If True (default) and filename is a list of files, sort them by BJD
            before reading
        progress : bool
            Show a progress bar while loading files
        """

        if '*' in file or '?' in file:
            file = glob(file)

        # list of files
        if isinstance(file, Iterable) and not isinstance(file, str):
            if progress:
                spectra = []
                for f in tqdm(file):
                    spectra.append(cls(f))
            else:
                spectra = [cls(f) for f in file]

            if sort_bjd:
                return sorted(spectra, key=lambda i: i.bjd)
            else:
                return spectra
        # just one file
        elif isinstance(file, str):
            return cls(file, **kwargs)
        # anything else
        else:
            raise ValueError(
                'Input to `from_file` should be a string or list of strings.')

    @cached_property
    def bjd(self):
        return getBJD(self.filename, mjd=False)

    def _check_for_s2d(self):
        if 'S1D' in self.filename:
            return exists(self.filename.replace('S1D', 'S2D'))
        elif 's1d' in self.filename:
            return exists(self.filename.replace('s1d', 'e2ds'))
        else:
            return False

    def _read_s2d(self):
        try:
            if 'S1D' in self.filename:
                return Spectrum2D(self.filename.replace('S1D', 'S2D'))
            elif 's1d' in self.filename:
                return Spectrum2D(self.filename.replace('s1d', 'e2ds'))
        except IndexError:
            return None

    def median_divide(self):
        if hasattr(self, '_median'):
            return
        self._median = np.median(self.flux)
        self.flux /= self._median

    def plot(self, ax=None, tellurics=True, flux_offset=0.0, legend=True):
        add_tellurics = tellurics and ax is None
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_figwidth(10)

        label = basename(self.filename).replace('.fits', '')
        ax.plot(self.wave[self.mask],
                self.flux[self.mask] + flux_offset,
                label=label)

        if add_tellurics:
            add_telluric_regions(ax)

        if legend:
            ax.legend()

        ax.set(xlabel=r'$\lambda$', ylabel='flux')
        plt.show()


class Spectrum2D:
    def __init__(self, filename):
        hdul = fits.open(filename)
        self.filename = filename
        self.HDU = hdul

        self.header = hdul[0].header

        inst = hdul[0].header['INSTRUME']
        if inst == 'ESPRESSO':
            self._SCIDATA = hdul[1].data
            self.flux = self._SCIDATA.copy()
            self._ERRDATA = hdul[2].data
            self._QUALDATA = hdul[3].data
            self.wave = self._WAVEDATA_VAC_BARY = hdul[4].data
            # self.wave = hdul[1].data['wavelength']
            # self.wave_air = hdul[1].data['wavelength_air']
            # self.flux = hdul[1].data['flux']
            # self.error = hdul[1].data['error']
            # self.quality = hdul[1].data['quality']
            # self.mask = self.quality == 0
        elif inst == 'HARPS':
            self._SCIDATA = hdul[0].data
            self.flux = self._SCIDATA.copy()
            norder, nwave = self.flux.shape
            self.wave = np.tile(np.arange(nwave), (norder, 1))

    def __repr__(self):
        r = f'Spectrum2D({basename(self.filename)})'
        return r

    def __len__(self):
        return 1

    # a convenience method to imitate the CCF constructor
    @classmethod
    def from_file(cls, file, hdu_number=1, data_index=-1, sort_bjd=True,
                  progress=True, **kwargs):
        """
        Create a `Spectrum` object from one or more fits files.

        Parameters
        ----------
        file : str or list of str
            The name(s) of the fits file(s)
        hdu_number : int, default = 1
            The index of the HDU list which contains the spectrum
        data_index : int, default = -1
            The index of the .data array which contains the CCF. The data will
            be accessed as ccf = HDU[hdu_number].data[data_index,:]
        sort_bjd : bool
            If True (default) and filename is a list of files, sort them by BJD
            before reading
        """

        if '*' in file or '?' in file:
            file = glob(file)

        # list of files
        if isinstance(file, Iterable) and not isinstance(file, str):
            if progress:
                spectra = []
                for f in tqdm(file):
                    spectra.append(cls(f))
            else:
                spectra = [cls(f) for f in file]

            if sort_bjd:
                return sorted(spectra, key=lambda i: i.bjd)
            else:
                return spectra
        # just one file
        elif isinstance(file, str):
            return cls(file, **kwargs)
        # anything else
        else:
            raise ValueError(
                'Input to `from_file` should be a string or list of strings.')

    @cached_property
    def bjd(self):
        return getBJD(self.filename, mjd=False)

    def median_divide(self):
        if hasattr(self, '_median'):
            return
        self._median = np.median(self._SCIDATA, axis=1)
        self.flux = (self.flux.T / self._median).T

    #     median = np.median(self.flux)
    #     self.flux /= median

    def plot(self, ax=None, tellurics=True, flux_offset=0.0, annotate=False):
        add_tellurics = tellurics and ax is None
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
        else:
            fig = ax.figure

        if annotate:
            match = {w: o for o, w in zip(np.arange(170), self.wave[:, 0])}
            annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)

            def update_annot(ind, line):
                x, y = line.get_data()
                order = match[x[0]]
                annot.xy = (x.mean(), y.max())
                text = f"order {order}"
                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(0.4)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    for line in lines:
                        cont, ind = line.contains(event)
                        if cont:
                            update_annot(ind, line)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                        else:
                            if vis:
                                annot.set_visible(False)
                                fig.canvas.draw_idle()

        lines = ax.plot(self.wave.T, self.flux.T + flux_offset)

        if add_tellurics:
            add_telluric_regions(ax)
        ax.set(xlabel=r'$\lambda$', ylabel='flux')

        if annotate:
            # fig.canvas.mpl_connect("motion_notify_event", hover)
            fig.canvas.mpl_connect('button_press_event', hover)

        plt.show()
