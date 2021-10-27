# import pytest


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
    from vera import DACE
    s = DACE.HD10180
    print(s)


def test_KOBE(capsys):
    from vera import KOBE

    # not in target list
    _ = KOBE.HD100
    cap = capsys.readouterr()
    assert cap.out == 'Cannot find "HD100" in KOBE target list.\n'

    # no access to data
    s = KOBE.KOBE_001
    assert s is None
