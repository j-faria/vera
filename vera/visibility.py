# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
from scipy.optimize import bisect
import datetime as dt
from dateutil import tz
import pickle
from random import choice
from PyAstronomy import pyasl
from astropy.coordinates import SkyCoord
from astropy.coordinates import name_resolve
from astropy.time import Time
from astropy import units
import ephem
import argparse
import calendar

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

import io
import matplotlib.pyplot as plt
import matplotlib
replace_figure = True
try:
    from PySide.QtGui import QApplication, QImage
except ImportError:
    try:
        from PyQt4.QtGui import QApplication, QImage
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QImage
        except ImportError:
            replace_figure = False


def add_clipboard_to_figures():
    # replace the original plt.figure() function with one that supports
    # clipboard-copying
    oldfig = plt.figure

    def newfig(*args, **kwargs):
        fig = oldfig(*args, **kwargs)

        def clipboard_handler(event):
            if event.key == 'ctrl+c':
                # store the image in a buffer using savefig(), this has the
                # advantage of applying all the default savefig parameters
                # such as background color; those would be ignored if you simply
                # grab the canvas using Qt
                buf = io.BytesIO()
                fig.savefig(buf)
                QApplication.clipboard().setImage(
                    QImage.fromData(buf.getvalue()))
                buf.close()
                print('Ctrl+C pressed: image is now in the clipboard')

        fig.canvas.mpl_connect('key_press_event', clipboard_handler)
        return fig

    plt.figure = newfig


if replace_figure:
    add_clipboard_to_figures()


def _parser():
    parser = argparse.ArgumentParser(
        description='Plot altitudes of objects against time for a specific night')

    parser.add_argument('targets', help='e.g. HD20010 or HD20010,HD41248',
                        nargs='+')

    parser.add_argument(
        '-d', '--date', default='today',
        help='Date in format YYYY-MM-DD (or YYYY if starobs). '
        'Default is today (this year if starobs).')

    parser.add_argument(
        '-P', '--period', default=None, type=str, nargs=1,
        help='Specify ESO period (October-March / April-September)')

    parser.add_argument(
        '-s', '--site', default='esolasilla',
        help='Observatory. Default is ESO La Silla. '
        'Common codes are esoparanal, lapalma, keck, lco, Palomar, etc')

    parser.add_argument(
        '-l', '--loc', default=None,
        help='Give the location of the observatory.'
        'Comma-separated altitude, latitude, longitude, timezone')

    parser.add_argument('-c', default=False, action='store_true',
                        help='Just print "target RA DEC" (to use in STARALT)')

    parser.add_argument(
        '-m', '--mode', choices=['staralt', 'starobs'], default='staralt',
        help='staralt: plot altitude against time for a particular night; '
        'starobs: plot how altitude changes over a year')

    parser.add_argument('--nomoon', default=False, action='store_true',
                        help="Don't plot moon altitude")

    parser.add_argument('--sh', default=None, type=float, nargs=1, dest='A',
                        help='Include plot of sunless hours above airmass A')

    parser.add_argument('--hover', default=False, action='store_true',
                        help='Color lines when mouse over')

    parser.add_argument(
        '-o', '--save', default=None, type=str, nargs=1,
        help='Save figure in output file (provide file extension)')

    parser.add_argument('--remove-watermark', default=False,
                        action='store_true',
                        help='Remove "Created with..." watermark text')

    parser.add_argument('--toi', default=False, action='store_true',
                        help='Targets are TOIs')

    return parser.parse_args()


def decdeg2dms(dd):
    """ Convert decimal degrees to deg,min,sec """
    is_positive = dd >= 0
    dd = abs(dd)
    minutes, seconds = divmod(dd * 3600, 60)
    degrees, minutes = divmod(minutes, 60)
    degrees = degrees if is_positive else -degrees
    return (degrees, minutes, seconds)


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


ESO_periods = {
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
}


def get_ESO_period(period):
    """ Return the JD of start and end of ESO period """
    assert isinstance(period, str) or isinstance(period, int)
    P = int(period)

    def getjd(y, m, d): return pyasl.jdcnv(dt.datetime(y, m, d))
    jd_start, jd_end = [getjd(*d) for d in ESO_periods[P]]

    return jd_start, jd_end


def hrs_up(up, down, eve, morn):
    """
    If an object comes up past a given point at `up`, and goes down at `down`, 
    and evening and morning are at `eve` and `morn`, computes how long object
    is up *and* it's dark.
    """
    # if any input is a float, assume it's JD
    if isinstance(up, float):
        up = pyasl.daycnv(up, mode='dt')
    if isinstance(down, float):
        down = pyasl.daycnv(down, mode='dt')
    if isinstance(eve, float):
        eve = pyasl.daycnv(eve, mode='dt')
    if isinstance(morn, float):
        morn = pyasl.daycnv(morn, mode='dt')

    SID_RATE = 1.0027379093
    if up < eve:
        if down >= morn:
            return (morn - eve).total_seconds() / 3600  # up all night
        elif down >= eve:
            # careful here ... circumpolar objects can come back *up* a second time
            # before morning.  up and down are the ones immediately preceding
            # and following the upper culmination nearest the center of the night,
            # so "up" can be on the previous night rather than the one we want. */
            up2 = up + dt.timedelta(days=1.0 / SID_RATE)
            if (up2 > morn):  # the usual case ... doesn't rise again
                return (down - eve).total_seconds() / 3600
            else:
                return ((down - eve) + (morn - up2)).total_seconds() / 3600
        else:
            return 0.
    elif down > morn:
        if up > morn:
            return 0.
        else:
            # again, a circumpolar object can be up at evening twilight and come
            # 'round again in the morning ...
            down0 = down - dt.timedelta(days=1.0 / SID_RATE)
            if down0 < eve:
                return (morn - up).total_seconds() / 3600
            else:
                return ((down0 - eve) + (morn - up)).total_seconds() / 3600
    else:
        return (down - up).total_seconds() / 3600
        # up & down the same night ... might happen a second time in pathological
        # cases, but this will be extremely rare except at very high latitudes.


SUN = ephem.Sun()


def get_next_sunset(jd, obs, mode='jd'):
    datetime_jd = pyasl.daycnv(jd, mode='dt')
    s = ephem.Observer()
    s.date = datetime_jd
    s.lat = ':'.join([str(i) for i in decdeg2dms(obs['latitude'])])
    s.lon = ':'.join([str(i) for i in decdeg2dms(obs['longitude'])])
    next_sunset = ephem.julian_date(s.next_setting(SUN))
    if mode == 'jd':
        return next_sunset
    elif mode == 'dt':
        return pyasl.daycnv(next_sunset, mode='dt')


def get_next_sunrise(jd, obs, mode='jd'):
    datetime_jd = pyasl.daycnv(jd, mode='dt')
    s = ephem.Observer()
    s.date = datetime_jd
    s.lat = ':'.join([str(i) for i in decdeg2dms(obs['latitude'])])
    s.lon = ':'.join([str(i) for i in decdeg2dms(obs['longitude'])])
    next_sunrise = ephem.julian_date(s.next_rising(SUN))
    if mode == 'jd':
        return next_sunrise
    elif mode == 'dt':
        return pyasl.daycnv(next_sunrise, mode='dt')


def get_next_pass_at_altitude(jd, altitude, target, obs, limit=0.25):
    """ Next time after jd that target passes at altitude, seen from obs """
    def alt(jd, target):
        ra = np.full_like(jd, target.ra.value)
        dec = np.full_like(jd, target.dec.value)
        lon, lat, alt = map(
            obs.__getitem__, ('longitude', 'latitude', 'altitude'))
        hor = pyasl.eq2hor(jd, ra, dec, lon=lon, lat=lat, alt=alt)
        return -altitude + hor[0]

    # if target is *already* above altitude at jd, return jd
    if alt(jd, target) > 0:
        return jd

    try:
        return bisect(alt, jd, jd + limit, args=(target, ))
    except ValueError:
        try:
            return bisect(alt, jd, jd + 2*limit, args=(target, ))
        except ValueError:
            return -99


def get_previous_pass_at_altitude(jd, altitude, target, obs, limit=0.25):
    """ 
    Previous time, before jd, that target passes at altitude, seen from obs 
    """
    def alt(jd, target):
        ra = np.full_like(jd, target.ra.value)
        dec = np.full_like(jd, target.dec.value)
        lon, lat, alt = map(obs.__getitem__,
                            ('longitude', 'latitude', 'altitude'))
        hor = pyasl.eq2hor(jd, ra, dec, lon=lon, lat=lat, alt=alt)
        return -altitude + hor[0]

    # if target is *still* above altitude at jd, return jd
    if alt(jd, target) > 0:
        return jd

    try:
        return bisect(alt, jd, jd - limit, args=(target, ))
    except ValueError:
        try:
            return bisect(alt, jd, jd - 2*limit, args=(target, ))
        except ValueError:
            return -99


def hrs_above_altitude(jd, altitude, target, obs):
    # evening
    eve = get_next_sunset(jd, obs)
    # star goes up (above altitude)
    up = get_next_pass_at_altitude(eve, altitude, target, obs)
    # print(eve, up)
    if up == -99:
        return 0.

    # morning
    morn = get_next_sunrise(jd, obs)
    if morn < eve:  # maybe of next day?
        morn = get_next_sunrise(jd+1, obs)
    # star goes down
    down = get_previous_pass_at_altitude(morn, altitude, target, obs)
    # print(morn, down)
    if down == -99:
        return 0.

    return hrs_up(up, down, eve, morn)


def get_visibility_curve(year, target, observatory, period=None):

    try:
        target = {'name': target, 'coord': SkyCoord.from_name(target)}
    except name_resolve.NameResolveError:
        print('Could not find target: {0!s}'.format(target))

    target_coord = target['coord']
    target_ra = target_coord.ra.deg
    target_dec = target_coord.dec.deg

    # set the observatory
    if isinstance(observatory, dict):
        obs = observatory
    else:
        obs = pyasl.observatory(observatory)

    if period is not None:
        jd_start, jd_end = get_ESO_period(period)
    else:
        jd_start = pyasl.jdcnv(dt.datetime(year, 1, 1))
        jd_end = pyasl.jdcnv(dt.datetime(year, 12, 31))

    jdbinsize = 1  # every day
    each_day = np.arange(jd_start, jd_end, jdbinsize)
    jds = []

    ## calculate the mid-dark times
    sun = ephem.Sun()
    for day in each_day:
        date_formatted = '/'.join([str(i) for i in pyasl.daycnv(day)[:-1]])
        s = ephem.Observer()
        s.date = date_formatted
        s.lat = ':'.join([str(i) for i in decdeg2dms(obs['latitude'])])
        s.lon = ':'.join([str(i) for i in decdeg2dms(obs['longitude'])])
        jds.append(ephem.julian_date(s.next_antitransit(sun)))
    jds = np.array(jds)

    # Get JD floating point
    jdsub = jds - np.floor(jds[0])

    # Get alt/az of object
    altaz = pyasl.eq2hor(jds, np.ones_like(jds)*target_ra, np.ones_like(jds)*target_dec,
                         lon=obs['longitude'], lat=obs['latitude'], alt=obs['altitude'])
    # plt.plot( jdsub, altaz[0], '-', color='k')

    return jds, altaz[0]


def StarObsPlot(year=None, targets=None, observatory=None, period=None,
                hover=False, sunless_hours=None, remove_watermark=False):
    """
    Plot the visibility of target.

    Parameters
    ----------
    year: int
        The year for which to calculate the visibility.
    targets: list
        List of targets.
        Each target should be a dictionary with keys 'name' and 'coord'.
        The key 'name' is a string, 'coord' is a SkyCoord object.
    observatory: string
        Name of the observatory that pyasl.observatory can resolve.
        Basically, any of pyasl.listObservatories().keys()
    period: string, optional
        ESO period for which to calculate the visibility. Overrides `year`.
    hover: boolean, optional
        If True, color visibility lines when mouse over.
    sunless_hours: float, optional
        If not None, plot sunless hours above this airmass
  """

    from mpl_toolkits.axes_grid1 import host_subplot
    from matplotlib.ticker import MultipleLocator
    from matplotlib.font_manager import FontProperties
    from matplotlib import rcParams
    rcParams['xtick.major.pad'] = 12
    font0 = FontProperties()
    font1 = font0.copy()
    font0.set_family('sans-serif')
    font0.set_weight('light')
    font1.set_family('sans-serif')
    font1.set_weight('medium')

    # set the observatory
    if isinstance(observatory, dict):
        obs = observatory
    else:
        obs = pyasl.observatory(observatory)

    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(left=0.07, right=0.8, bottom=0.15, top=0.88)

    # watermak
    if not remove_watermark:
        fig.text(0.99, 0.99,
                 'Created with\ngithub.com/iastro-pt/ObservationTools',
                 fontsize=10, color='gray', ha='right', va='top', alpha=0.5)

    # plotting sunless hours?
    shmode = False
    if sunless_hours is not None:
        shmode = True
        # limit in airmass (assumed plane-parallel atm)
        shairmass = sunless_hours
        # correspoing limit in altitude
        def f(alt): return pyasl.airmassPP(alt) - shairmass
        shalt = 90 - bisect(f, 0, 89)

    if shmode:
        fig.subplots_adjust(hspace=0.35)
        ax = host_subplot(211)
        axsh = host_subplot(212)
        plt.text(0.5, 0.47,
                 "- sunless hours above airmass {:.1f} - \n".format(shairmass),
                 transform=fig.transFigure, ha='center', va='bottom',
                 fontsize=12)
        plt.text(0.5, 0.465,
                 "the thick line above the curves represents the total sunless hours "
                 "for each day of the year",
                 transform=fig.transFigure, ha='center', va='bottom', fontsize=10)

    else:
        ax = host_subplot(111)

    for n, target in enumerate(targets):

        target_coord = target['coord']
        target_ra = target_coord.ra.deg
        target_dec = target_coord.dec.deg

        if period is not None:
            jd_start, jd_end = get_ESO_period(period)
        else:
            jd_start = pyasl.jdcnv(dt.datetime(year, 1, 1))
            jd_end = pyasl.jdcnv(dt.datetime(year, 12, 31))

        jdbinsize = 1  # every day
        each_day = np.arange(jd_start, jd_end, jdbinsize)
        jds = []

        ## calculate the mid-dark times
        sun = ephem.Sun()
        for day in each_day:
            date_formatted = '/'.join([str(i) for i in pyasl.daycnv(day)[:-1]])
            s = ephem.Observer()
            s.date = date_formatted
            s.lat = ':'.join([str(i) for i in decdeg2dms(obs['latitude'])])
            s.lon = ':'.join([str(i) for i in decdeg2dms(obs['longitude'])])
            jds.append(ephem.julian_date(s.next_antitransit(sun)))

        jds = np.array(jds)

        # Get JD floating point
        jdsub = jds - np.floor(jds[0])

        # Get alt/az of object
        altaz = pyasl.eq2hor(jds, np.ones_like(jds)*target_ra, np.ones_like(jds)*target_dec,
                             lon=obs['longitude'], lat=obs['latitude'], alt=obs['altitude'])
        ax.plot(jdsub, altaz[0], '-', color='k')

        # label for each target
        plabel = "[{0:2d}]  {1!s}".format(n + 1, target['name'])

        # number of target at the top of the curve
        ind_label = np.argmax(altaz[0])
        # or at the bottom if the top is too close to the corners
        # if jdsub[ind_label] < 5 or jdsub[ind_label] > jdsub.max()-5:
        #   ind_label = np.argmin(altaz[0])
        ax.text(jdsub[ind_label], altaz[0][ind_label], str(n+1), color="b", fontsize=14,
                fontproperties=font1, va="bottom", ha="center")

        if n + 1 == 29:
            # too many?
            ax.text(1.1, 1.0-float(n+1)*0.04, "too many targets", ha="left", va="top", transform=ax.transAxes,
                    fontsize=10, fontproperties=font0, color="r")
        else:
            ax.text(1.1, 1.0-float(n+1)*0.04, plabel, ha="left", va="top", transform=ax.transAxes,
                    fontsize=12, fontproperties=font0, color="b")

    if shmode:
        sunless_hours = []
        for day in each_day:
            date_formatted = '/'.join([str(i) for i in pyasl.daycnv(day)[:-1]])
            s = ephem.Observer()
            s.date = date_formatted
            s.lat = ':'.join([str(i) for i in decdeg2dms(obs['latitude'])])
            s.lon = ':'.join([str(i) for i in decdeg2dms(obs['longitude'])])
            # hours from sunrise to sunset
            td = pyasl.daycnv(ephem.julian_date(s.next_setting(sun)), mode='dt') \
                - pyasl.daycnv(ephem.julian_date(s.next_rising(sun)), mode='dt')
            sunless_hours.append(24 - td.total_seconds() / 3600)

        days = each_day - np.floor(each_day[0])
        axsh.plot(days, sunless_hours, '-', color='k', lw=2)
        axsh.set(
            ylim=(0, 15), yticks=range(1, 15), ylabel='Useful hours',
            yticklabels=[r'${}^{{\rm h}}$'.format(n) for n in range(1, 15)])

    ax.text(1.1, 1.03, "List of targets", ha="left", va="top", transform=ax.transAxes,
            fontsize=12, fontproperties=font0, color="b")

    axrange = ax.get_xlim()

    if period is None:
        months = range(1, 13)
        ndays = [0] + [calendar.monthrange(year, m)[1] for m in months]
        ax.set_xlim([0, 366])
        ax.set_xticks(np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
        ax.set_xticklabels(
            map(calendar.month_abbr.__getitem__, months), fontsize=10)
        if shmode:
            axsh.set_xlim([0, 366])
            axsh.set_xticks(np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
            axsh.set_xticklabels(
                map(calendar.month_abbr.__getitem__, months), fontsize=10)
    else:
        if int(period) % 2 == 0:
            # even ESO period, Oct -> Mar
            months = [10, 11, 12, 1, 2, 3]
            ndays = [0] + [calendar.monthrange(year, m)[1] for m in months]
            ax.set_xlim([0, 181])
            ax.set_xticks(np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
            ax.set_xticklabels(
                map(calendar.month_abbr.__getitem__, months), fontsize=10)
            if shmode:
                axsh.set_xlim([0, 181])
                axsh.set_xticks(
                    np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
                axsh.set_xticklabels(
                    map(calendar.month_abbr.__getitem__, months), fontsize=10)
        else:
            # odd ESO period, Apr -> Sep
            months = range(4, 10)
            ndays = [0] + [calendar.monthrange(year, m)[1] for m in months]
            ax.set_xlim([0, 182])
            ax.set_xticks(np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
            ax.set_xticklabels(
                map(calendar.month_abbr.__getitem__, months), fontsize=10)
            if shmode:
                axsh.set_xlim([0, 182])
                axsh.set_xticks(
                    np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
                axsh.set_xticklabels(
                    map(calendar.month_abbr.__getitem__, months), fontsize=10)

    if axrange[1] - axrange[0] <= 1.0:
        jdhours = np.arange(0, 3, 1.0 / 24.)
        utchours = (np.arange(0, 72, dtype=int) + 12) % 24
    else:
        jdhours = np.arange(0, 3, 1.0 / 12.)
        utchours = (np.arange(0, 72, 2, dtype=int) + 12) % 24

    # Make ax2 responsible for "top" axis and "right" axis
    ax2 = ax.twin()
    # Set upper x ticks
    ax2.set_xticks(np.cumsum(ndays))
    ax2.set_xlabel("Day")

    # plane-parallel airmass
    airmass_ang = np.arange(10, 81, 5)
    geo_airmass = pyasl.airmass.airmassPP(airmass_ang)[::-1]
    ax2.set_yticks(airmass_ang)
    airmassformat = []
    for t in range(geo_airmass.size):
        airmassformat.append("{0:2.2f}".format(geo_airmass[t]))
    ax2.set_yticklabels(airmassformat)  # , rotation=90)
    ax2.set_ylabel("Relative airmass", labelpad=32)
    ax2.tick_params(axis="y", pad=6, labelsize=8)
    plt.text(1.02, -0.04, "Plane-parallel", transform=ax.transAxes, ha='left',
             va='top', fontsize=10, rotation=90)

    ax22 = ax.twin()
    ax22.set_xticklabels([])
    ax22.set_frame_on(True)
    ax22.patch.set_visible(False)
    ax22.yaxis.set_ticks_position('right')
    ax22.yaxis.set_label_position('right')
    ax22.spines['right'].set_position(('outward', 30))
    ax22.spines['right'].set_color('k')
    ax22.spines['right'].set_visible(True)
    airmass2 = list(
        map(
            lambda ang: pyasl.airmass.airmassSpherical(
                90. - ang, obs['altitude']),
            airmass_ang))
    ax22.set_yticks(airmass_ang)
    airmassformat = []
    for t in range(len(airmass2)):
        airmassformat.append(" {0:2.2f}".format(airmass2[t]))
    ax22.set_yticklabels(airmassformat, rotation=90)
    ax22.tick_params(axis="y", pad=8, labelsize=8)
    plt.text(1.05, -0.04, "Spherical+Alt", transform=ax.transAxes, ha='left', va='top',
             fontsize=10, rotation=90)

    ax.set_ylim([0, 91])
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    yticks = ax.get_yticks()
    ytickformat = []
    for t in range(yticks.size):
        ytickformat.append(str(int(yticks[t])) + r"$^\circ$")
    ax.set_yticklabels(ytickformat, fontsize=11 if shmode else 16)
    ax.set_ylabel("Altitude", fontsize=18)
    yticksminor = np.array(ax.get_yticks(minor=True))
    ymind = np.where(yticksminor % 15. != 0.)[0]
    yticksminor = yticksminor[ymind]
    ax.set_yticks(yticksminor, minor=True)
    m_ytickformat = []
    for t in range(yticksminor.size):
        m_ytickformat.append(str(int(yticksminor[t])) + r"$^\circ$")
    ax.set_yticklabels(m_ytickformat, minor=True)
    ax.set_ylim([0, 91])

    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', which="minor", linestyle='dotted')
    ax2.xaxis.grid(color='gray', linestyle='dotted')

    if period is not None:
        plt.text(
            0.5, 0.95,
            "Visibility over P{0!s}\n - altitudes at mid-dark time -".format(
                period), transform=fig.transFigure, ha='center', va='bottom',
            fontsize=12)
    else:
        plt.text(
            0.5, 0.95,
            "Visibility over {0!s}\n - altitudes at mid-dark time -".format(
                year), transform=fig.transFigure, ha='center', va='bottom',
            fontsize=12)

    obsco = "Obs coord.: {0:8.4f}$^\circ$, {1:8.4f}$^\circ$, {2:4f} m".format(
        obs['longitude'], obs['latitude'], obs['altitude'])

    plt.text(0.01, 0.97, obsco, transform=fig.transFigure, ha='left',
             va='center', fontsize=10)
    plt.text(0.01, 0.95, obs['name'], transform=fig.transFigure, ha='left',
             va='center', fontsize=10)

    # interactive!
    if hover:
        main_axis = fig.axes[0]
        all_lines = set(main_axis.get_lines())

        def on_plot_hover(event):
            for line in main_axis.get_lines():
                if line.contains(event)[0]:
                    line.set_color('red')  # make this line red
                    # and all others black
                    all_other_lines = all_lines - set([line])
                    for other_line in all_other_lines:
                        other_line.set_color('black')
                    fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

    return fig


def StarObsAxis(ax, year=None, targets=None, observatory=None, period=None,
                hover=False, sunless_hours=None, remove_watermark=False):
    """
    Plot the visibility of target.

    Parameters
    ----------
    year: int
        The year for which to calculate the visibility.
    targets: list
        List of targets.
        Each target should be a dictionary with keys 'name' and 'coord'.
        The key 'name' is a string, 'coord' is a SkyCoord object.
    observatory: string
        Name of the observatory that pyasl.observatory can resolve.
        Basically, any of pyasl.listObservatories().keys()
    period: string, optional
        ESO period for which to calculate the visibility. Overrides `year`.
    hover: boolean, optional
        If True, color visibility lines when mouse over.
    sunless_hours: float, optional
        If not None, plot sunless hours above this airmass
  """

    from mpl_toolkits.axes_grid1 import host_subplot
    from matplotlib.ticker import MultipleLocator
    from matplotlib.font_manager import FontProperties
    from matplotlib import rcParams
    # rcParams['xtick.major.pad'] = 12
    font0 = FontProperties()
    font1 = font0.copy()
    font0.set_family('sans-serif')
    font0.set_weight('light')
    font1.set_family('sans-serif')
    font1.set_weight('medium')

    # set the observatory
    if isinstance(observatory, dict):
        obs = observatory
    else:
        obs = pyasl.observatory(observatory)

    # fig = plt.figure(figsize=(15, 10))
    # fig.subplots_adjust(left=0.07, right=0.8, bottom=0.15, top=0.88)

    # watermak
    # if not remove_watermark:
    #     fig.text(0.99, 0.99,
    #              'Created with\ngithub.com/iastro-pt/ObservationTools',
    #              fontsize=10, color='gray', ha='right', va='top', alpha=0.5)

    # plotting sunless hours?
    shmode = False
    if sunless_hours is not None:
        shmode = True
        # limit in airmass (assumed plane-parallel atm)
        shairmass = sunless_hours
        # correspoing limit in altitude
        def f(alt): return pyasl.airmassPP(alt) - shairmass
        shalt = 90 - bisect(f, 0, 89)

    if shmode:
        fig.subplots_adjust(hspace=0.35)
        ax = host_subplot(211)
        axsh = host_subplot(212)
        plt.text(0.5, 0.47,
                 "- sunless hours above airmass {:.1f} - \n".format(shairmass),
                 transform=fig.transFigure, ha='center', va='bottom',
                 fontsize=12)
        plt.text(0.5, 0.465,
                 "the thick line above the curves represents the total sunless hours "
                 "for each day of the year",
                 transform=fig.transFigure, ha='center', va='bottom', fontsize=10)

    for n, target in enumerate(targets):

        target_coord = target['coord']
        target_ra = target_coord.ra.deg
        target_dec = target_coord.dec.deg

        if period is not None:
            jd_start, jd_end = get_ESO_period(period)
        else:
            jd_start = pyasl.jdcnv(dt.datetime(year, 1, 1))
            jd_end = pyasl.jdcnv(dt.datetime(year, 12, 31))

        jdbinsize = 1  # every day
        each_day = np.arange(jd_start, jd_end, jdbinsize)
        jds = []

        ## calculate the mid-dark times
        sun = ephem.Sun()
        for day in each_day:
            date_formatted = '/'.join([str(i) for i in pyasl.daycnv(day)[:-1]])
            s = ephem.Observer()
            s.date = date_formatted
            s.lat = ':'.join([str(i) for i in decdeg2dms(obs['latitude'])])
            s.lon = ':'.join([str(i) for i in decdeg2dms(obs['longitude'])])
            jds.append(ephem.julian_date(s.next_antitransit(sun)))

        jds = np.array(jds)

        # Get JD floating point
        jdsub = jds - np.floor(jds[0])

        # Get alt/az of object
        altaz = pyasl.eq2hor(jds, np.ones_like(jds)*target_ra, np.ones_like(jds)*target_dec,
                             lon=obs['longitude'], lat=obs['latitude'], alt=obs['altitude'])
        ax.plot(jdsub, altaz[0], '-', color='k', lw=0.8)
        ax.plot(jdsub[altaz[0] > 30], altaz[0]
                [altaz[0] > 30], '-', color='g', lw=2)

        # label for each target
        # plabel = "[{0:2d}]  {1!s}".format(n + 1, target['name'])

        # # number of target at the top of the curve
        # ind_label = np.argmax(altaz[0])
        # # or at the bottom if the top is too close to the corners
        # # if jdsub[ind_label] < 5 or jdsub[ind_label] > jdsub.max()-5:
        # #   ind_label = np.argmin(altaz[0])
        # ax.text( jdsub[ind_label], altaz[0][ind_label], str(n+1), color="b", fontsize=14, \
        #          fontproperties=font1, va="bottom", ha="center")

        # if n + 1 == 29:
        #     # too many?
        #     ax.text(1.1, 1.0-float(n+1)*0.04, "too many targets", ha="left", va="top", transform=ax.transAxes, \
        #             fontsize=10, fontproperties=font0, color="r")
        # else:
        #     ax.text(1.1, 1.0-float(n+1)*0.04, plabel, ha="left", va="top", transform=ax.transAxes, \
        #             fontsize=12, fontproperties=font0, color="b")

    if shmode:
        sunless_hours = []
        for day in each_day:
            date_formatted = '/'.join([str(i) for i in pyasl.daycnv(day)[:-1]])
            s = ephem.Observer()
            s.date = date_formatted
            s.lat = ':'.join([str(i) for i in decdeg2dms(obs['latitude'])])
            s.lon = ':'.join([str(i) for i in decdeg2dms(obs['longitude'])])
            # hours from sunrise to sunset
            td = pyasl.daycnv(ephem.julian_date(s.next_setting(sun)), mode='dt') \
                - pyasl.daycnv(ephem.julian_date(s.next_rising(sun)), mode='dt')
            sunless_hours.append(24 - td.total_seconds() / 3600)

        days = each_day - np.floor(each_day[0])
        axsh.plot(days, sunless_hours, '-', color='k', lw=2)
        axsh.set(
            ylim=(0, 15), yticks=range(1, 15), ylabel='Useful hours',
            yticklabels=[r'${}^{{\rm h}}$'.format(n) for n in range(1, 15)])

    # ax.text(1.1, 1.03, "List of targets", ha="left", va="top", transform=ax.transAxes, \
    #         fontsize=12, fontproperties=font0, color="b")

    axrange = ax.get_xlim()

    if period is None:
        months = range(1, 13)
        ndays = [0] + [calendar.monthrange(year, m)[1] for m in months]
        ax.set_xlim([0, 366])
        ax.set_xticks(np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
        ax.set_xticklabels(
            map(calendar.month_abbr.__getitem__, months), fontsize=8)
        # if shmode:
        #     axsh.set_xlim([0, 366])
        #     axsh.set_xticks(np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
        #     axsh.set_xticklabels(
        #         map(calendar.month_abbr.__getitem__, months), fontsize=10)
    else:
        if int(period) % 2 == 0:
            # even ESO period, Oct -> Mar
            months = [10, 11, 12, 1, 2, 3]
            ndays = [0] + [calendar.monthrange(year, m)[1] for m in months]
            ax.set_xlim([0, 181])
            ax.set_xticks(np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
            ax.set_xticklabels(
                map(calendar.month_abbr.__getitem__, months), fontsize=10)
            if shmode:
                axsh.set_xlim([0, 181])
                axsh.set_xticks(
                    np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
                axsh.set_xticklabels(
                    map(calendar.month_abbr.__getitem__, months), fontsize=10)
        else:
            # odd ESO period, Apr -> Sep
            months = range(4, 10)
            ndays = [0] + [calendar.monthrange(year, m)[1] for m in months]
            ax.set_xlim([0, 182])
            ax.set_xticks(np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
            ax.set_xticklabels(
                map(calendar.month_abbr.__getitem__, months), fontsize=10)
            if shmode:
                axsh.set_xlim([0, 182])
                axsh.set_xticks(
                    np.cumsum(ndays)[:-1] + (np.array(ndays) / 2.)[1:])
                axsh.set_xticklabels(
                    map(calendar.month_abbr.__getitem__, months), fontsize=10)

    if axrange[1] - axrange[0] <= 1.0:
        jdhours = np.arange(0, 3, 1.0 / 24.)
        utchours = (np.arange(0, 72, dtype=int) + 12) % 24
    else:
        jdhours = np.arange(0, 3, 1.0 / 12.)
        utchours = (np.arange(0, 72, 2, dtype=int) + 12) % 24

    ax.vlines(np.cumsum(ndays)[:-1], 0, 90, color='k', alpha=0.2)
    ax.hlines([30], 0, 366, lw=0.8)
    ax.vlines(dt.datetime.now().timetuple().tm_yday, 30, 90, color='b')

    # Make ax2 responsible for "top" axis and "right" axis
    ax2 = ax.twinx()
    # Set upper x ticks
    # ax2.xaxis.tick_top()
    # ax2.set_xticks(np.cumsum(ndays))
    # ax2.set_xlabel("Day")
    # print(ax.get_xlim())

    # plane-parallel airmass
    airmass_ang = np.arange(0, 81, 5)
    geo_airmass = pyasl.airmass.airmassPP(airmass_ang)[::-1]
    ax2.set_yticks(airmass_ang)
    airmassformat = []
    for t in range(geo_airmass.size):
        airmassformat.append("{0:2.2f}".format(geo_airmass[t]))
    ax2.set_yticklabels(airmassformat)  # , rotation=90)
    ax2.set_ylabel("Relative airmass", labelpad=5)
    ax2.tick_params(axis="y", pad=6, labelsize=8)
    ax2.set_ylim(-9, 80)
    # plt.text(1.02,-0.04, "Plane-parallel", transform=ax.transAxes, ha='left', \
    #          va='top', fontsize=10, rotation=90)

    # ax22 = ax.twinx()
    # ax22.set_xticklabels([])
    # ax22.set_frame_on(True)
    # ax22.patch.set_visible(False)
    # ax22.yaxis.set_ticks_position('right')
    # ax22.yaxis.set_label_position('right')
    # ax22.spines['right'].set_position(('outward', 30))
    # ax22.spines['right'].set_color('k')
    # ax22.spines['right'].set_visible(True)
    # airmass2 = list(
    #     map(
    #         lambda ang: pyasl.airmass.airmassSpherical(90. - ang, obs['altitude']),
    #         airmass_ang))
    # ax22.set_yticks(airmass_ang)
    # airmassformat = []
    # for t in range(len(airmass2)):
    #     airmassformat.append(" {0:2.2f}".format(airmass2[t]))
    # ax22.set_yticklabels(airmassformat, rotation=90)
    # ax22.tick_params(axis="y", pad=8, labelsize=8)
    # plt.text(1.05,-0.04, "Spherical+Alt", transform=ax.transAxes, ha='left', va='top', \
    #          fontsize=10, rotation=90)

    ax.set_ylim([0, 90])
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    yticks = ax.get_yticks()
    ytickformat = []
    for t in range(yticks.size):
        ytickformat.append(str(int(yticks[t])) + r"$^\circ$")
    ax.set_yticklabels(ytickformat, fontsize=10)
    ax.set_ylabel("Altitude", fontsize=10)
    yticksminor = ax.get_yticks(minor=True)
    # ymind = np.where(yticksminor % 15. != 0.)[0]
    # yticksminor = yticksminor[ymind]
    # ax.set_yticks(yticksminor, minor=True)
    # m_ytickformat = []
    # for t in range(yticksminor.size):
    #     m_ytickformat.append(str(int(yticksminor[t])) + r"$^\circ$")
    # ax.set_yticklabels(m_ytickformat, minor=True)
    ax.set_ylim([0, 90])

    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', which="minor", linestyle='dotted')
    ax2.xaxis.grid(color='gray', linestyle='dotted')

    # if period is not None:
    #     plt.text(
    #         0.5, 0.95,
    #         "Visibility over P{0!s}\n - altitudes at mid-dark time -".format(
    #             period), transform=fig.transFigure, ha='center', va='bottom',
    #         fontsize=12)
    # else:
    #     plt.text(
    #         0.5, 0.95,
    #         "Visibility over {0!s}\n - altitudes at mid-dark time -".format(
    #             year), transform=fig.transFigure, ha='center', va='bottom',
    #         fontsize=12)

    obsco = "Obs coord.: {0:8.4f}$^\circ$, {1:8.4f}$^\circ$, {2:.0f} m".format(
        obs['longitude'], obs['latitude'], obs['altitude'])

    ax.set_title(obsco, loc='left', fontsize=6)
    ax.set_title('Altitudes at mid-dark time', loc='right', fontsize=8)

    # plt.text(0.01, 0.97, obsco, transform=fig.transFigure, ha='left',
    #          va='center', fontsize=10)
    # plt.text(0.01, 0.95, obs['name'], transform=fig.transFigure, ha='left',
    #          va='center', fontsize=10)

    # interactive!
    if hover:
        main_axis = fig.axes[0]
        all_lines = set(main_axis.get_lines())

        def on_plot_hover(event):
            for line in main_axis.get_lines():
                if line.contains(event)[0]:
                    line.set_color('red')  # make this line red
                    # and all others black
                    all_other_lines = all_lines - set([line])
                    for other_line in all_other_lines:
                        other_line.set_color('black')
                    fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

    # return fig


def VisibilityPlot(date=None, targets=None, observatory=None, plotLegend=True,
                   showMoon=True, showMoonDist=True, print2file=False,
                   remove_watermark=False):
    """
    Plot the visibility of target.

    Parameters
    ----------
    date: datetime
        The date for which to calculate the visibility.
    targets: list
        List of targets.
        Each target should be a dictionary with keys 'name' and 'coord'.
        The key 'name' is aa string, 'coord' is a SkyCoord object.
    observatory: string
        Name of the observatory that pyasl.observatory can resolve.
        Basically, any of pyasl.listObservatories().keys()
    plotLegend: boolean, optional
        If True (default), show a legend.
    showMoonDist : boolean, optional
        If True (default), the Moon distance will be shown.
  """

    from mpl_toolkits.axes_grid1 import host_subplot
    from matplotlib.ticker import MultipleLocator
    from matplotlib.font_manager import FontProperties
    from matplotlib import rcParams
    rcParams['xtick.major.pad'] = 12

    if isinstance(observatory, dict):
        obs = observatory
    else:
        obs = pyasl.observatory(observatory)

    # observer = ephem.Observer()
    # observer.pressure = 0
    # observer.horizon = '-0:34'
    # observer.lat, observer.lon = obs['latitude'], obs['longitude']
    # observer.date = date
    # print(observer.date)
    # print(observer.previous_rising(ephem.Sun()))
    # print(observer.next_setting(ephem.Sun()))
    # print(observer.previous_rising(ephem.Moon()))
    # print(observer.next_setting(ephem.Moon()))
    # observer.horizon = '-6'
    # noon = observer.next_transit(ephem.Sun())
    # print(noon)
    # print(observer.previous_rising(ephem.Sun(), start=noon, use_center=True))
    # print()

    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(left=0.07, right=0.8, bottom=0.15, top=0.88)

    # watermak
    if not remove_watermark:
        fig.text(0.99, 0.99,
                 'Created with\ngithub.com/iastro-pt/ObservationTools',
                 fontsize=10, color='gray', ha='right', va='top', alpha=0.5)

    ax = host_subplot(111)

    font0 = FontProperties()
    font1 = font0.copy()
    font0.set_family('sans-serif')
    font0.set_weight('light')
    font1.set_family('sans-serif')
    font1.set_weight('medium')

    for n, target in enumerate(targets):

        target_coord = target['coord']
        target_ra = target_coord.ra.deg
        target_dec = target_coord.dec.deg

        # JD array
        jdbinsize = 1.0 / 24. / 20.
        # jds = np.arange(allData[n]["Obs jd"][0], allData[n]["Obs jd"][2], jdbinsize)
        jd = pyasl.jdcnv(date)
        jd_start = pyasl.jdcnv(date) - 0.5
        jd_end = pyasl.jdcnv(date) + 0.5
        jds = np.arange(jd_start, jd_end, jdbinsize)
        # Get JD floating point
        jdsub = jds - np.floor(jds[0])
        # Get alt/az of object
        altaz = pyasl.eq2hor(jds, np.ones(jds.size)*target_ra, np.ones(jds.size)*target_dec,
                             lon=obs['longitude'], lat=obs['latitude'], alt=obs['altitude'])
        # Get alt/az of Sun
        sun_position = pyasl.sunpos(jd)
        sun_ra, sun_dec = sun_position[1], sun_position[2]
        sunpos_altaz = pyasl.eq2hor(jds, np.ones(jds.size)*sun_ra, np.ones(jds.size)*sun_dec,
                                    lon=obs['longitude'], lat=obs['latitude'], alt=obs['altitude'])

        # Define plot label
        plabel = "[{0:2d}]  {1!s}".format(n + 1, target['name'])

        # Find periods of: day, twilight, and night
        day = np.where(sunpos_altaz[0] >= 0.)[0]
        twi = np.where(
            np.logical_and(sunpos_altaz[0] > -18., sunpos_altaz[0] < 0.))[0]
        night = np.where(sunpos_altaz[0] <= -18.)[0]

        if (len(day) == 0) and (len(twi) == 0) and (len(night) == 0):
            print
            print("VisibilityPlot - no points to draw")
            print

        if showMoon:
            # plot the moon
            mpos = pyasl.moonpos(jds)
            # mpha = pyasl.moonphase(jds)
            mpos_altaz = pyasl.eq2hor(jds, mpos[0], mpos[1],
                                      lon=obs['longitude'],
                                      lat=obs['latitude'], alt=obs['altitude'])
            ax.plot(jdsub, mpos_altaz[0], color='k', alpha=0.3, ls='--',
                    label='Moon')
            # moonind = np.where( mpos_altaz[0] > 0. )[0]

            if showMoonDist:
                mdist = pyasl.getAngDist(mpos[0], mpos[1], np.ones(jds.size)*target_ra,
                                         np.ones(jds.size)*target_dec)
                bindist = int((2.0 / 24.) / jdbinsize)
                firstbin = np.random.randint(0, bindist)
                for mp in range(0, int(len(jds) / bindist)):
                    bind = firstbin + mp * bindist
                    if altaz[0][bind] - 1. < 5.:
                        continue
                    ax.text(jdsub[bind], altaz[0][bind]-1., str(int(mdist[bind]))+r"$^\circ$", ha="center", va="top",
                            fontsize=8, stretch='ultra-condensed', fontproperties=font0, alpha=1.)

        if len(twi) > 1:
            # There are points in twilight
            linebreak = np.where(
                (jdsub[twi][1:] - jdsub[twi][:-1]) > 2.0 * jdbinsize)[0]
            if len(linebreak) > 0:
                plotrjd = np.insert(jdsub[twi], linebreak + 1, np.nan)
                plotdat = np.insert(altaz[0][twi], linebreak + 1, np.nan)
                ax.plot(plotrjd, plotdat, "-", color='#BEBEBE', linewidth=1.5)
            else:
                ax.plot(jdsub[twi], altaz[0][twi], "-", color='#BEBEBE',
                        linewidth=1.5)

        ax.plot(jdsub[night], altaz[0][night], '.k', label=plabel)
        ax.plot(jdsub[day], altaz[0][day], '.', color='#FDB813')

        altmax = np.argmax(altaz[0])
        ax.text(jdsub[altmax], altaz[0][altmax], str(n+1), color="b", fontsize=14,
                fontproperties=font1, va="bottom", ha="center")

        if n + 1 == 29:
            ax.text(1.1, 1.0-float(n+1)*0.04, "too many targets", ha="left", va="top", transform=ax.transAxes,
                    fontsize=10, fontproperties=font0, color="r")
        else:
            ax.text(1.1, 1.0-float(n+1)*0.04, plabel, ha="left", va="top", transform=ax.transAxes,
                    fontsize=12, fontproperties=font0, color="b")

    ax.text(1.1, 1.03, "List of targets", ha="left", va="top", transform=ax.transAxes,
            fontsize=12, fontproperties=font0, color="b")

    axrange = ax.get_xlim()
    ax.set_xlabel("UT [hours]")

    if axrange[1] - axrange[0] <= 1.0:
        jdhours = np.arange(0, 3, 1.0 / 24.)
        utchours = (np.arange(0, 72, dtype=int) + 12) % 24
    else:
        jdhours = np.arange(0, 3, 1.0 / 12.)
        utchours = (np.arange(0, 72, 2, dtype=int) + 12) % 24
    ax.set_xticks(jdhours)
    ax.set_xlim(axrange)
    ax.set_xticklabels(utchours, fontsize=18)

    # Make ax2 responsible for "top" axis and "right" axis
    ax2 = ax.twin()
    # Set upper x ticks
    ax2.set_xticks(jdhours)
    ax2.set_xticklabels(utchours, fontsize=18)
    ax2.set_xlabel("UT [hours]")

    # Horizon angle for airmass
    airmass_ang = np.arange(5., 90., 5.)
    geo_airmass = pyasl.airmass.airmassPP(90. - airmass_ang)
    ax2.set_yticks(airmass_ang)
    airmassformat = []
    for t in range(geo_airmass.size):
        airmassformat.append("{0:2.2f}".format(geo_airmass[t]))
    ax2.set_yticklabels(airmassformat, rotation=90)
    ax2.set_ylabel("Relative airmass", labelpad=32)
    ax2.tick_params(axis="y", pad=10, labelsize=10)
    plt.text(1.015, -0.04, "Plane-parallel", transform=ax.transAxes, ha='left',
             va='top', fontsize=10, rotation=90)

    ax22 = ax.twin()
    ax22.set_xticklabels([])
    ax22.set_frame_on(True)
    ax22.patch.set_visible(False)
    ax22.yaxis.set_ticks_position('right')
    ax22.yaxis.set_label_position('right')
    ax22.spines['right'].set_position(('outward', 25))
    ax22.spines['right'].set_color('k')
    ax22.spines['right'].set_visible(True)
    airmass2 = list(
        map(
            lambda ang: pyasl.airmass.airmassSpherical(
                90. - ang, obs['altitude']),
            airmass_ang))
    ax22.set_yticks(airmass_ang)
    airmassformat = []
    for t in airmass2:
        airmassformat.append("{0:2.2f}".format(t))
    ax22.set_yticklabels(airmassformat, rotation=90)
    ax22.tick_params(axis="y", pad=10, labelsize=10)
    plt.text(1.045, -0.04, "Spherical+Alt", transform=ax.transAxes, ha='left', va='top',
             fontsize=10, rotation=90)

    ax3 = ax.twiny()
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    ax3.xaxis.set_ticks_position('bottom')
    ax3.xaxis.set_label_position('bottom')
    ax3.spines['bottom'].set_position(('outward', 50))
    ax3.spines['bottom'].set_color('k')
    ax3.spines['bottom'].set_visible(True)

    ltime, ldiff = pyasl.localtime.localTime(
        utchours, np.repeat(obs['longitude'], len(utchours)))
    jdltime = jdhours - ldiff / 24.
    ax3.set_xticks(jdltime)
    ax3.set_xticklabels(utchours)
    ax3.set_xlim([axrange[0], axrange[1]])
    ax3.set_xlabel("Local time [hours]")

    ax.set_ylim([0, 91])
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    yticks = ax.get_yticks()
    ytickformat = []
    for t in range(yticks.size):
        ytickformat.append(str(int(yticks[t])) + r"$^\circ$")
    ax.set_yticklabels(ytickformat, fontsize=16)
    ax.set_ylabel("Altitude", fontsize=18)
    yticksminor = ax.get_yticks(minor=True)
    ymind = np.where(yticksminor % 15. != 0.)[0]
    yticksminor = yticksminor[ymind]
    ax.set_yticks(yticksminor, minor=True)
    m_ytickformat = []
    for t in range(yticksminor.size):
        m_ytickformat.append(str(int(yticksminor[t])) + r"$^\circ$")
    ax.set_yticklabels(m_ytickformat, minor=True)
    ax.set_ylim([0, 91])

    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', which="minor", linestyle='dotted')
    ax2.xaxis.grid(color='gray', linestyle='dotted')

    plt.text(0.5, 0.95, "Visibility on {0!s}".format(date.date()),
             transform=fig.transFigure, ha='center', va='bottom', fontsize=20)

    if plotLegend:
        line1 = matplotlib.lines.Line2D((0, 0), (1, 1), color='#FDB813',
                                        linestyle="-", linewidth=2)
        line2 = matplotlib.lines.Line2D((0, 0), (1, 1), color='#BEBEBE',
                                        linestyle="-", linewidth=2)
        line3 = matplotlib.lines.Line2D((0, 0), (1, 1), color='k',
                                        linestyle="-", linewidth=2)
        line4 = matplotlib.lines.Line2D((0, 0), (1, 1), color='k', alpha=0.2,
                                        linestyle="--", linewidth=2)

        if showMoon:
            lgd2 = plt.legend((line1, line2, line3, line4),
                              ("day", "twilight", "night", "Moon"),
                              bbox_to_anchor=(0.88, 0.18), loc='best',
                              borderaxespad=0, prop={'size': 12}, fancybox=True)
        else:
            lgd2 = plt.legend((line1, line2, line3),
                              ("day", "twilight", "night"),
                              bbox_to_anchor=(0.88, 0.18), loc='best',
                              borderaxespad=0, prop={'size': 12}, fancybox=True)

        lgd2.get_frame().set_alpha(.9)

    obsco = r"Obs coord.: {0:8.4f}$^\circ$, {1:8.4f}$^\circ$, {2:4.2f} m"
    obsco = obsco.format(obs['longitude'], obs['latitude'], obs['altitude'])

    plt.text(0.01, 0.97, obsco, transform=fig.transFigure, ha='left',
             va='center', fontsize=10)
    plt.text(0.01, 0.95, obs['name'], transform=fig.transFigure, ha='left',
             va='center', fontsize=10)

    return fig


if __name__ == '__main__':
    args = _parser()

    target_names = args.targets[0].split(',')

    ## Get coordinates for all the targets
    targets = []

    # flush keyword was not backported to Python < 3.3
    if sys.version_info[:2] < (3, 3):
        print('Sending queries to CDS...', end=' ')
        sys.stdout.flush()
    else:
        print('Sending queries to CDS...', end=' ', flush=True)

    for target_name in tqdm(target_names):
        if args.toi:  # check the table
            # data = np.genfromtxt('TOI-info.csv', delimiter=',', names=True)
            # data = np.loadtxt('TOI-info.csv', delimiter=',', usecols=(1, 16,17), skiprows=1, dtype={'names': ('TOI', 'RA', 'Dec'), 'formats': (np.float, '|S15', '|S15')},)
            data = np.loadtxt('TOI-info.csv', delimiter=',', usecols=(1, 15, 16),
                              skiprows=1, dtype={'names': ('TOI', 'RA', 'Dec'), 'formats': 3*[float]})
            ind = np.where(data['TOI'].astype(int) == int(target_name))[0]

            if ind.size == 0:
                print('Could not find target: {0!s}'.format(target_name))
                continue

            ind = ind[0]
            coord = SkyCoord(data[ind]['RA'], data[ind]['Dec'], unit=units.deg)

            targets.append({
                'name': target_name,
                'coord': CacheSkyCoord(coord)
            })
        else:
            try:
                targets.append({
                    'name': target_name,
                    'coord': CacheSkyCoord.from_name(target_name)
                })
            except name_resolve.NameResolveError as e:
                print('Could not find target: {0!s}'.format(target_name))

    ## Just print coordinates in STARALT format and exit
    if args.c:
        print('Coordinates for {0!s}\n'.format(args.targets[0]))
        for target in targets:
            ## name hh mm ss ±dd mm ss
            out = '{0!s}'.format(target['name'])
            ra = target['coord'].ra.hms
            out += ' {0:02d} {1:02d} {2:5.3f}'.format(
                int(ra.h), int(ra.m), ra.s)
            dec = target['coord'].dec.dms
            out += ' {0:02d} {1:02d} {2:5.3f}'.format(
                int(dec.d), int(dec.m), dec.s)
            print(out)

        sys.exit(0)

    ## Actually calculate the visibility curves
    print('Calculating visibility for {0!s}'.format(args.targets[0]))

    P = args.period
    if args.period is not None:
        if args.mode != 'starobs':
            print('Specifying ESO period is only possible in "starobs" mode')
            sys.exit(1)

        P = args.period[0]
        P = P.replace('P', '')  # if user gave --period P100, for example

    if args.date == 'today':
        if args.mode == 'staralt':
            # now() gives the current *time* which we don't want
            today = dt.datetime.now()
            date = dt.datetime(today.year, today.month, today.day,
                               tzinfo=tz.tzutc())
        elif args.mode == 'starobs':
            date = dt.datetime.now().year
    else:
        if args.mode == 'staralt':
            if "-" not in args.date:
                raise ValueError(
                    "Date needs to be provided as YYYY-MM-DD for staralt mode."
                )
            ymd = [int(i) for i in args.date.split('-')]
            date = dt.datetime(*ymd)
        elif args.mode == 'starobs':
            if "-" in args.date:
                date = int(args.date.split('-')[0])
            else:
                date = int(args.date)

    ## Find observatory
    if args.loc is None:
        available_sites = pyasl.listObservatories(show=False)

        if args.site.lower() in ('paranal', 'vlt', 'UT1', 'UT2', 'UT3', 'UT4'):
            args.site = 'esoparanal'

        if args.site.lower() not in available_sites.keys():
            print('"{0!s}" is not a valid observatory code. '
                  'Try one of the following:\n'.format(args.site)
                  )

            maxCodeLen = max(map(len, available_sites.keys()))
            print(("{0:" + str(maxCodeLen) + "s}     ").format("Code") +
                  "Observatory name")
            print("-" * (21 + maxCodeLen))
            for k in sorted(available_sites.keys(), key=lambda s: s.lower()):
                print(("{0:" + str(maxCodeLen) + "s} --- ").format(k) +
                      available_sites[k]["name"])

            sys.exit(1)
        site = args.site

    else:
        loc = list(map(float, args.loc.split(',')))
        site = {
            'altitude': loc[0],
            'latitude': loc[1],
            'longitude': loc[2],
            'tz': loc[3],
            'name': 'unknown'
        }

    if args.mode == 'staralt':
        fig = VisibilityPlot(date=date, targets=targets, observatory=site,
                             remove_watermark=args.remove_watermark,
                             showMoon=not args.nomoon)

    elif args.mode == 'starobs':
        if args.A is not None:
            am = args.A[0]
        else:
            am = None

        fig = StarObsPlot(year=date, targets=targets, observatory=site,
                          period=P, hover=args.hover, sunless_hours=am,
                          remove_watermark=args.remove_watermark)

    if args.save is not None:
        print('Saving the figure to {}'.format(args.save[0]))
        fig.savefig(args.save[0])
    else:
        plt.show()
