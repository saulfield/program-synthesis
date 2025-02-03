def make_counter():
    i = 0

    def f():
        nonlocal i
        r = i
        i += 1
        return r

    return f


_gen_str_id = make_counter()
_ = _gen_str_id()
_str_id_map: dict[str, int] = dict()


def str_to_id(s: str) -> int:
    id = _str_id_map.get(s)
    if id is None:
        id = _gen_str_id()
        _str_id_map[s] = id
    return id
