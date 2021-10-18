import sys
import os
import warnings
import pickle
from difflib import SequenceMatcher
from astroquery.simbad import Simbad

cSimbad = Simbad()
cSimbad.add_votable_fields('ids', 'sptype', 'flux(V)', 'flux(B)')

greek_CAP = ('Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta',
             'Theta', 'Iota', 'Kappa', 'Lamda', 'Mu', 'Nu', 'Xi', 'Omicron',
             'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi',
             'Omega')

greek_SMA = tuple(l.lower() for l in greek_CAP)

# Alphabetical listing of constellations
constellations = [
    'Andromeda',
    'Antlia',
    'Apus',
    'Aquarius',
    'Aquila',
    'Ara',
    'Aries',
    'Auriga',
    'Boötes',
    'Caelum',
    'Camelopardalis',
    'Cancer',
    'Canes Venatici',
    'Canis Major',
    'Canis Minor',
    'Capricornus',
    'Carina',
    'Cassiopeia',
    'Centaurus',
    'Cepheus',
    'Cetus',
    'Chamaeleon',
    'Circinus',
    'Columba',
    'Coma Berenices',
    'Corona Austrina',
    'Corona Borealis',
    'Corvus',
    'Crater',
    'Crux',
    'Cygnus',
    'Delphinus',
    'Dorado',
    'Draco',
    'Equuleus',
    'Eridanus',
    'Fornax',
    'Gemini',
    'Grus',
    'Hercules',
    'Horologium',
    'Hydra',
    'Hydrus',
    'Indus',
    'Lacerta',
    'Leo',
    'Leo Minor',
    'Lepus',
    'Libra',
    'Lupus',
    'Lynx',
    'Lyra',
    'Mensa',
    'Microscopium',
    'Monoceros',
    'Musca',
    'Norma',
    'Octans',
    'Ophiuchus',
    'Orion',
    'Pavo',
    'Pegasus',
    'Perseus',
    'Phoenix',
    'Pictor',
    'Pisces',
    'Piscis Austrinus',
    'Puppis',
    'Pyxis',
    'Reticulum',
    'Sagitta',
    'Sagittarius',
    'Scorpius',
    'Sculptor',
    'Scutum',
    'Serpens',
    'Sextans',
    'Taurus',
    'Telescopium',
    'Triangulum',
    'Triangulum Australe',
    'Tucana',
    'Ursa Major',
    'Ursa Minor',
    'Vela',
    'Virgo',
    'Volans',
    'Vulpecula ',
]

this_dir = os.path.dirname(os.path.abspath(__file__))
constellation_file = os.path.join(this_dir, 'constellations.txt')
if os.path.exists(constellation_file):
    constellations_abbrev = [l.strip().split()[0] 
                            for l in open(constellation_file).readlines()]
    const_abb = tuple(constellations_abbrev)
else:
    const_abb = ()


def _lcsubstr(s1, s2):
    """ Find the length of the longest common substring between s1 and s2 """
    SM = SequenceMatcher(None, s1, s2)
    return SM.find_longest_match(0, len(s1), 0, len(s2)).size


def _de_escape(star):
    if 'TOI-' in star:
        return star.replace('TOI-', 'TOI ')
    return star


def get_vmag(star):
    star = _de_escape(star)
    try:
        Vm = pickle.load(open('Vmags.pickle', 'rb'))
    except FileNotFoundError:
        Vm = {}
    
    try:
        return Vm[star]
    except KeyError:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vmag = cSimbad.query_object(star)['FLUX_V'][0]
            Vm[star] = vmag
            pickle.dump(Vm, open('Vmags.pickle', 'wb'), protocol=-1)
            return vmag
        except TypeError:
            raise ValueError(f'Cannot find {star} in Simbad')


def get_bmv(star):
    star = _de_escape(star)
    try:
        bmv = pickle.load(open('BmV.pickle', 'rb'))
    except FileNotFoundError:
        bmv = {}
    
    try:
        return bmv[star]
    except KeyError:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vmag = cSimbad.query_object(star)['FLUX_V'][0]
                bmag = cSimbad.query_object(star)['FLUX_B'][0]
            bmv[star] = bmag - vmag
            pickle.dump(bmv, open('BmV.pickle', 'wb'), protocol=-1)
            return bmv[star]
        except TypeError:
            raise ValueError(f'Cannot find {star} in Simbad')


def getSPtype(star):
    star = _de_escape(star)
    try:
        STs = pickle.load(open('STs.pickle', 'rb'))
    except FileNotFoundError:
        STs = {}

    try:
        return STs[star]
    except KeyError:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    ST = cSimbad.query_object(star)['SP_TYPE'][0].decode()
                except AttributeError:
                    ST = cSimbad.query_object(star)['SP_TYPE'][0]
            STs[star] = ST
            pickle.dump(STs, open('STs.pickle', 'wb'), protocol=-1)
            return ST
        except TypeError:
            return ''


def getIDs(star, remove_space=False, show=False, allnames=False):
    star = _de_escape(star)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        query = cSimbad.query_object(star)

    if query is None:
        raise ValueError(f'Cannot find {star} in Simbad')

    assert 'IDS' in query.colnames, f'{query}'
    try:
        IDS = query['IDS'][0].split('|')
    except TypeError:
        IDS = query['IDS'][0].decode().split('|')

    starIDs = IDS

    if allnames:
        otherIDs = starIDs
    else:
        known = ('HD', 'GJ', 'LHS', 'Gl', 'Ross', 'NAME', 'Wolf') \
                + greek_SMA + const_abb
        otherIDs = [ID for ID in starIDs if any([kn in ID for kn in known])]

    otherIDs = [ID.replace('V* ', '') for ID in otherIDs]
    otherIDs = [ID.replace('* ', '') for ID in otherIDs]
    otherIDs = [ID.replace('.0', '') for ID in otherIDs]
    otherIDs = [ID.replace('NAME ', '') for ID in otherIDs]
    otherIDs = [ID.replace("'s", '') for ID in otherIDs]
    otherIDs = [ID.replace("Star", '') for ID in otherIDs]
    otherIDs = [ID.replace("star", '') for ID in otherIDs]

    if remove_space:
        sep = ''
    else:
        sep = ' '

    otherIDs = [sep.join(ID.split()) for ID in otherIDs]

    # ugly hack!
    if 'Proxima' in otherIDs:
        try:
            otherIDs.remove('Proxima Centauri')
            otherIDs.remove('ProximaCentauri')
        except ValueError:
            pass
        try:
            otherIDs.remove('Proxima')
        except ValueError:
            pass
        # keep 'Proxima Cen'

    # remove the name of the star itself    
    try:
        otherIDs.remove(star)
    except ValueError:
        pass
    # and similar ones
    otherIDs = [ID for ID in otherIDs if (star not in ID and _lcsubstr(star,ID) < 3)]

    otherIDs = list(set(otherIDs))


    if show:
        otherIDs = ['   ' + ID for ID in otherIDs]
        print(star)
        print('\n'.join(otherIDs))
        return

    return otherIDs


def getIDs_startswith(star, string):
    ids = getIDs(star, allnames=True)
    return [i for i in ids if i.startswith(string)][0]


if __name__ == "__main__":
    try:
        star = sys.argv[1]
    except IndexError:
        sys.exit(0)

    if star == 'all':
        stars = [line.strip() for line in open('list_stars.txt')]
        [getIDs(star, show=True) for star in stars]
        sys.exit(0)

    otherIDs = getIDs(star)

    otherIDs = ['   ' + ID for ID in otherIDs]

    print(star)
    print('\n'.join(otherIDs))
