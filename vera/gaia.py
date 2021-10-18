import warnings
import requests
# import signal
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

from .query_simbad import _de_escape

as_yr = u.arcsec / u.year
mas_yr = u.milliarcsecond / u.year
mas = u.milliarcsecond

aSimbad = Simbad()
aSimbad.add_votable_fields('rv_value', 'pm', 'pmdec', 'pmra', 'plx', 'ra(d)', 'dec(d)')


def handler(signum, frame):
    print('Timeout!')
    raise Exception


def build_query(star):
    star = _de_escape(star)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = aSimbad.query_object(star)
    if r is None:
        raise ValueError(f'star {star} not found in Simbad')
    else:
        r = r[0]

    query = f"""SELECT TOP 1 * FROM gaiadr2.gaia_source \
    WHERE CONTAINS(
                POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),
                CIRCLE('ICRS',
                    COORD1(EPOCH_PROP_POS({r['RA_d']},{r['DEC_d']},{r['PLX_VALUE']},{r['PMRA']},{r['PMDEC']},{r['RV_VALUE']},2000,2015.5)),
                    COORD2(EPOCH_PROP_POS({r['RA_d']},{r['DEC_d']},{r['PLX_VALUE']},{r['PMRA']},{r['PMDEC']},{r['RV_VALUE']},2000,2015.5)),
                    0.001388888888888889))=1
    """

    return r, query


def run_job(query, timeout=5):
    # signal.signal(signal.SIGALRM, handler)
    # signal.alarm(timeout)
    job = Gaia.launch_job(query)
    return job.get_results()


def mu(pmra, pmdec):
    return np.sqrt(pmra**2 + pmdec**2)


def secular_acceleration(star, verbose=False):
    try:
        table = Gaia.query_object(star, radius=0.5 * u.arcsec)
        if len(table) > 0:
            μα = (table[0]['pmra'] * mas_yr)
            μδ = (table[0]['pmdec'] * mas_yr)
            π = (table[0]['parallax'] * mas)
        else:
            raise NameResolveError

    # except NameResolveError:
    except Exception:
        rsimbad, query = build_query(star)

        try:
            r = run_job(query)[0]
            μα = (r['pmra'] * mas_yr)
            μδ = (r['pmdec'] * mas_yr)
            π = (r['parallax'] * mas)
        # except requests.HTTPError:
        except Exception:
            π = 0.0

        if π == 0.0:
            print(f"Warning: Couldn't get the GAIA parallax for {star}. Using Simbad.")
            μα = (rsimbad['PMRA'] * mas_yr)
            μδ = (rsimbad['PMDEC'] * mas_yr)
            π = (rsimbad['PLX_VALUE'] * mas)

    d = π.to(u.pc, equivalencies=u.parallax())

    if verbose:
        print(f'  d: {d:.2f}')
        print(f'  μα: {μα:.4f}')
        print(f'  μδ: {μδ:.4f}')

    μ = μα**2 + μδ**2
    sa = (μ * d).to(u.m / u.second / u.year,
                    equivalencies=u.dimensionless_angles())

    return sa.value
