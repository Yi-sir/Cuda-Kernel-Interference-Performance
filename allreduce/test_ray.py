import ray


@ray.remote
def f(a, b, c):
    return a + b + c


object_ref = f.remote(1, 2, 3)
result = ray.get(object_ref)

assert result == (1 + 2 + 3)
