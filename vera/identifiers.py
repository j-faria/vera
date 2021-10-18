import sys
from astroquery.simbad import Simbad

cSimbad = Simbad()
cSimbad.add_votable_fields('ids')

greek_CAP = ('Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta',
             'Theta', 'Iota', 'Kappa', 'Lamda', 'Mu', 'Nu', 'Xi', 'Omicron',
             'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi',
             'Omega')

greek_SMA = tuple(l.lower() for l in greek_CAP)

def getIDs(star, remove_space=False, show=False, allnames=False):
    query = cSimbad.query_object(star)

    if query is None:
        raise ValueError(f'Cannot find {star}')

    assert 'IDS' in query.colnames, f'{query}'
    starIDs = query['IDS'][0].decode().split('|')

    if allnames:
        otherIDs = starIDs
    else:
        known = ('HD', 'GJ', 'LHS', 'Gl') + greek_SMA
        otherIDs = [ID for ID in starIDs if any([kn in ID for kn in known])]

    otherIDs = [ID.replace('* ', '') for ID in otherIDs]
    otherIDs = [ID.replace('.0', '') for ID in otherIDs]

    if remove_space:
        sep = ''
    else:
        sep = ' '

    otherIDs = [sep.join(ID.split()) for ID in otherIDs]

    try:
        otherIDs.remove(star)
    except ValueError:
        pass

    otherIDs = list(set(otherIDs))

    if show:
        otherIDs = ['   ' + ID for ID in otherIDs]
        print(star)
        print('\n'.join(otherIDs))
        return

    return otherIDs


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