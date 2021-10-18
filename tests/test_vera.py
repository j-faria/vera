def test_imports():
    import vera
    from vera import RV


def test_dace():
    from vera.query_dace import get_observations
    _ = get_observations('HD10180', verbose=False)


def test_read_rdb():
    from vera import RV
    s = RV('data_file.rdb', sigmaclip=False)
    print(s)
