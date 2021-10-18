def test_imports():
    import vera


def test_dace():
    from vera.query_dace import get_observations
    _ = get_observations('HD10180', verbose=False)
