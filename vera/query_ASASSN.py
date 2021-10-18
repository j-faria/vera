    # utf8=✓&ra=12.28088633&dec=57.81580444&radius=0.1&vmag_min=&vmag_max=&epochs_min=&epochs_max=&rms_min=&rms_max=&sort_by=raj2000


import os
import sys
from datetime import datetime
import requests
from bs4 import BeautifulSoup

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.timeseries import LombScargle
from astroquery.simbad import Simbad

# cSimbad = Simbad()
# cSimbad.add_votable_fields('ids', 'sptype', 'flux(V)', 'flux(B)')

main_url = 'https://asas-sn.osu.edu/photometry?'
export_csv_url = 'https://asas-sn.osu.edu/photometry/export_source/%s?type=csv'


def build_search_url(star, ra=None, dec=None, limit=1, radius=0.5):
    if ra is None or dec is None:
        print('querying object with Simbad...')
        q = Simbad.query_object(star)
        ra = q['RA'][0].replace(' ', ':')
        dec = q['DEC'][0].replace(' ', ':').replace('+', '')

    url = f'{main_url}'
    url += f'ra={ra}&dec={dec}&radius={radius}'
    url += '&vmag_min=&vmag_max=&epochs_min=&epochs_max=&rms_min=&rms_max=&sort_by=raj2000'
    return url


def query(star, ra=None, dec=None):
    url = build_search_url(star, ra, dec)
    response = requests.get(url)
    if response.status_code != 200:
        print(response)
        return
    return response.content.decode()


def parse(content):
    if 'No objects matching specified criteria were found.' in content:
        raise ValueError('superWASP query did not match any object')

    soup = BeautifulSoup(content, 'html.parser')

    total_found = soup.find('div', attrs={'class': 'query-total'}).text
    total_found = int(total_found.strip().split('\n')[1])
    if total_found == 0:
        raise ValueError('ASAS-SN query did not match any object')

    tablerow = soup.find('table').find_all('tr')[1]
    data = np.array(tablerow.find_all('td'), dtype=object)

    inds = [0, 4]
    name, epochs = data[inds]

    source = name.a.attrs['href'].replace('/photometry/', '')
    csv_link = export_csv_url % source
    name = name.text.strip()
    epochs = int(epochs.text)
    return csv_link, name, epochs


def get_lightcurve(star, verbose=True):
    content = query(star)
    csv_link, name, epochs = parse(content)
    if verbose:
        print(f'Found "{name}" ({epochs} observations)')

    filename = name.replace(' ', '_') + '.csv'
    if os.path.exists(filename):
        return filename

    # download the lightcurve
    if verbose:
        print('Downloading lightcurve...', end=' ', flush=True)
    url = csv_link
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


if __name__ == "__main__":
    get_lightcurve(sys.argv[1])