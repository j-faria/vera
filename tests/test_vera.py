def test_imports():
    import vera
    from vera import RV


def test_dace():
    from vera.query_dace import get_observations
    _ = get_observations('HD10180', verbose=False)


def test_read_rdb():
    from vera import RV
    from os.path import dirname, join
    here = dirname(__file__)
    s = RV(join(here, 'data_file.rdb'), star='dummy', sigmaclip=False)
    print(s)


def test_DACE():
    from os.path import exists
    from vera import DACE
    s = DACE.HD10180
    print(s)
