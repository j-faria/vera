""" 
Query the ESPRESSO exposure time calculator
"""

import re
import requests
from itertools import product
import datetime
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from astropy.time import Time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

url = 'http://archive.eso.org/wdb/wdb/eso/sched_rep_arc/query'
#?wdbo=html%2fdisplay&max_rows_returned=100&tab_period=on&period=P102&tab_obs_mode=on&tab_run_type=on&run_type=%25&start=&tab_nights=on&tel=%25&tab_instrument=on&instrument=%25&instrument_user=&progid=1102.C-0744&tab_pi_coi=on&pi_coi=&pi_coi_name=PI_only&title=&tab_dp_id=on&tab_publications=on&order=DATE_ASC&

form_data = {
    'wdbo': 'csv',
    'top': '1000',
    'period': 'P102',
    'run_type': 'GTO',
    'tel': '%',
    'instrument': 'ESPRESSO',
    # 'progid': '1102.C-0744',
    'pi_coi_name': 'PI_only',
    'order': 'DATE_ASC',
}


ESO_periods = {
    # 108: [(2021, 10, 1), (2022, 3, 31)],
    # 107: [(2021, 4, 1), (2021, 9, 30)],
    106: [(2020, 10, 1), (2021, 3, 31)],
    105: [(2020, 4, 1), (2020, 9, 30)],
    104: [(2019, 10, 1), (2020, 3, 31)],
    103: [(2019, 4, 1), (2019, 9, 30)],
    102: [(2018, 10, 1), (2019, 3, 31)],
    101: [(2018, 4, 1), (2018, 9, 30)],
    100: [(2017, 10, 1), (2018, 3, 31)],
    99: [(2017, 4, 1), (2017, 9, 30)],
    98: [(2016, 10, 1), (2017, 3, 31)],
    97: [(2016, 4, 1), (2016, 9, 30)],
    96: [(2015, 10, 1), (2016, 3, 31)],
    95: [(2015, 4, 1), (2015, 9, 30)],
    94: [(2014, 10, 1), (2015, 3, 31)],
    93: [(2014, 4, 1), (2014, 9, 30)],
    92: [(2013, 10, 1), (2014, 3, 31)],
    91: [(2013, 4, 1), (2013, 9, 30)],
    90: [(2012, 10, 1), (2013, 3, 31)],
    89: [(2012, 4, 1), (2012, 9, 30)],
    88: [(2011, 10, 1), (2012, 3, 31)],
    87: [(2011, 4, 1), (2011, 9, 30)],
    86: [(2010, 10, 1), (2011, 3, 31)],
    85: [(2010, 4, 1), (2010, 9, 30)],
    84: [(2009, 10, 1), (2010, 3, 31)],
}


for k,v in ESO_periods.items():
    ESO_periods[k].append([Time('-'.join(map(str, vv))).jd - 24e5 for vv in v])



def parse_date(date):
    month = {
        'Jan': '01',
        'Feb': '02',
        'Mar': '03',
        'Apr': '04',
        'May': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12',
    }
    return '-'.join([date[-4:], month[date[3:6]], date[:2]])


def query(period='102', debug=False):
    """ Query the ESO page for the GTO runs """
    fill = form_data
    fill['period'] = 'P' + period
    r = requests.post(url, data=fill)
    if debug:
        print(r.text)
    return r


def get_JDs(period='102', night=True, arrays=True, verbose=True):
    """ 
    Get the Julian days for all ESPRESSO GTO runs in a given period. If
    `night`=True, return the JD of sunset and sunrise. This function returns the
    runs' start and end in arrays (lists if `arrays`=False).
    """
    if night:
        # from astroplan import Observer
        # paranal = Observer.at_site("paranal")
        import astroobs as obs
        VLT = obs.Observation('vlt', moonAvoidRadius=15, horizon_obs=0)

    if isinstance(period, int):
        period = str(period)
    if ',' in period:
        periods = period.split(',')
    else:
        periods = [period]

    starts, ends = [], []
    for period in periods:
        if verbose:
            print(f'Period: {period},', end=' ')
            print('starting ESO query...', end=' ', flush=True)

        r = query(period)

        if verbose:
            print('done')

        lines = r.text.split('\n')[2:-1]
        pattern = re.compile(r"between \d* \w* \d* and \d* \w* \d*")

        if verbose and night:
            print('calculating sunset/sunrise times...')

        for line in lines:
            try:
                found = re.findall(pattern, line)[0]
            except IndexError:
                continue

            date1 = found[8:-16]
            if night:
                t = Time(parse_date(date1) + ' 12:00:00')
                VLT.change_date(t.datetime)
                jd1 = Time(datetime.datetime.strptime(str(VLT.sunset),
                           r'%Y/%m/%d %H:%M:%S')).mjd
                # jd1 = paranal.sun_set_time(t, 'next').mjd
            else:
                jd1 = Time(parse_date(date1)).mjd # - 24e5

            date2 = found[24:]
            if night:
                t = Time(parse_date(date2) + ' 12:00:00')
                VLT.change_date(t.datetime)
                jd2 = Time(datetime.datetime.strptime(str(VLT.sunset),
                           r'%Y/%m/%d %H:%M:%S')).mjd
                # jd2 = paranal.sun_rise_time(t, 'previous').mjd
            else:
                jd2 = Time(parse_date(date2)).mjd # - 24e5

            starts.append(jd1)
            ends.append(jd2)

    starts, ind = np.unique(starts, return_index=True)
    ends = np.array(ends)[ind]

    if arrays:
        return starts, ends
    else:
        return list(starts), list(ends)


def plot_runs(starts=None, ends=None, period='102', night=True, today=True, **kwargs):
    """ Make a simple plot with the allocated runs """
    if starts is None and ends is None:
        starts, ends = get_JDs(period, night=night)

    fig, ax = plt.subplots(1, 1)
    for start, end in zip(starts, ends):
        # if start > today:
        ax.axvspan(start, end, **kwargs)
    if today:
        now = Time.now().jd - 24e5
        ax.axvline(x=now, color='k', lw=2)
    ax.set(xlabel='MJD [days]', yticklabels=[], yticks=[])

    return starts, ends




def get_random_times(N, periods, seed=None, mjd=True, return_available=False):
    raise NotImplementedError
