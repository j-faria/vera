import os
from glob import glob
import numpy as np

try:
    from actin.actin import actin as ACTIN  # actinception
    _actin_available = True
except ImportError:
    _actin_available = False

from .utils import info

def actin(self):
    if not _actin_available:
        print('Please (pip) install actin to use this function')
        return

    # download and load S2D spectra
    self.enable_spectra(s1d=False)

    # call actin
    output_dir = os.path.join(self.spec_dir, 'actin')
    indices = ['I_CaII', 'I_NaI', 'I_Ha16', 'I_Ha06', 'I_HeI', 'I_CaI']

    if all([hasattr(self, index) for index in indices]):
        if self.verbose:
            info('actin alread calculated')
        return

    ACTIN(files=[s.filename for s in self.spectra],
          calc_index=indices,
          obj_name=self.star,
          save_output=output_dir,
          del_out=False,
          verbose=False)

    results = load_actin_results(output_dir, indices)
    for index, v in results.items():
        # add attribute to self
        setattr(self, index, v)
        # add attribute to self.each
        for i, individual in enumerate(self.each):
            m = self.obs == i + 1
            setattr(individual, index, v[m])
        # keep track of attributes in _activity_indicators and print
        if not index.endswith('_err'):
            self._activity_indicators.add(index)
            if self.verbose:
                info(f'Adding {index} and {index}_err as attributes')



def load_actin_results(directory, indices):
    rdb = glob(os.path.join(directory, '*/*.rdb'))[0]
    data = np.genfromtxt(rdb,
                         names=True,
                         comments='--',
                         dtype=None,
                         encoding=None)
    r = {}
    for i in indices:
        r.update(as_dict(data[[i, i + '_err']]))
    return r


def as_dict(rec):
    """ Turn a numpy recarray record into a dict. """
    return {name: rec[name] for name in rec.dtype.names}
