import sys
import os
# import warnings
# import pickle
from datetime import datetime
import requests
from bs4 import BeautifulSoup

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.timeseries import LombScargle
# from astroquery.simbad import Simbad

# cSimbad = Simbad()
# cSimbad.add_votable_fields('ids', 'sptype', 'flux(V)', 'flux(B)')

main_url = 'https://wasp.cerit-sc.cz'


def build_search_url(star, limit=1, radius=1):
    star = star.replace(' ', '').replace('+', '%2B')
    url = f'{main_url}/search?'\
          f'objid={star}&limit={limit}&radius={radius}&radiusUnit=deg'
    return url


def query(star):
    url = build_search_url(star)
    response = requests.get(url)
    if response.status_code != 200:
        print(response)
        return
    return response.content.decode()


def parse(content, star):
    if 'not found in Sesame' in content:
        raise ValueError(f'object ID "{star}" not found')

    if 'No objects matching specified criteria were found.' in content:
        raise ValueError('superWASP query did not match any object')

    soup = BeautifulSoup(content, 'html.parser')
    table = soup.find_all('table')[0]
    tablerow = table.find_all('tr')[1]
    data = np.array(tablerow.find_all('td'), dtype=object)

    inds = [2, 3, 4, 6, 7, 8, 9, 10]
    name, npts, files, start, stop, ra, dec, mag = data[inds]

    name = name.text
    npts = int(npts.text)
    csv_link = files.find_all('a')[1].attrs['href']

    start, stop = start.text, stop.text
    start = datetime.strptime(start[:-2], '%Y-%m-%d %H:%M:%S')
    stop = datetime.strptime(stop[:-2], '%Y-%m-%d %H:%M:%S')

    ra, dec = float(ra.text), float(dec.text)
    mag = float(mag.text)

    return name, npts, csv_link, start, stop, ra, dec, mag


def get_lightcurve(star, verbose=True):
    content = query(star)
    name, npts, csv_link, start, stop, ra, dec, mag = parse(content, star)

    if verbose:
        print(
            f'Found "{name}" ({npts} observations '
            f'between {start.date()} and {stop.date()})'
        )

    filename = name.replace(' ', '_') + '.csv'

    if os.path.exists(filename):
        return filename

    # download the lightcurve
    if verbose:
        print('Downloading lightcurve...', end=' ', flush=True)
    url = main_url + csv_link
    response = requests.get(url)
    if response.status_code != 200:
        if verbose:
            print('failed!')
        return

    # save the lightcurve to a file
    with open(filename, 'w') as f:
        f.write(response.text)

    if verbose:
        print()
        print(f'Saved lightcurve to {filename}')

    return filename


class superWASP:
    def __init__(self, filename, verbose=True):
        self.verbose = verbose

        # read the lightcurve
        data = np.genfromtxt(filename, delimiter=',', names=True)
        self.target = filename[:-4]
        self.N = data.size
        self.time = data['HJD']
        self.time -= 24e5
        self.mag = data['magnitude']
        median_mag = np.median(self.mag)
        self.flux = np.negative(self.mag - median_mag) + median_mag
        self.flux_err = data['magnitude_error']

        self.c_time = self.time.copy()
        self.c_flux = self.flux.copy()
        self.c_flux_err = self.flux_err.copy()
        self.mask = np.ones_like(self.flux, dtype=bool)

    def __repr__(self):
        return f'superWASP({self.target}, {self.N} points)'

    @classmethod
    def query_object(cls, star, verbose=True):
        filename = get_lightcurve(star, verbose)
        return cls(filename, verbose=verbose)

    def sigmaclip(self, start_sigma=4, step=0.8, niter=5):
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
            plotit(original, mask)
            go_on = input(msg.format(sigma))
            if go_on == 's':
                break
            sigma *= step

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
                                          alpha=0.6, ecolor='k')
        [bar.set_alpha(0.2) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        ax.set(ylabel='-mag', xlabel='JD [days]')
        return ax, fig

    def gls(self):
        model = LombScargle(self.c_time, self.c_flux, self.c_flux_err)
        f, p = model.autopower()
        if (p < 0).any():
            f, p = model.autopower(method='cython')
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        ax.semilogx(1/f, p)
        # ax.hlines(model.false_alarm_level([0.01, 0.1]), *ax.get_xlim(),
        #           color='k', alpha=0.5)
        return model


if __name__ == "__main__":
    get_lightcurve(sys.argv[1])
