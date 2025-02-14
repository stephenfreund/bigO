import bigO


def fact(x: int) -> int:
    v = 1
    for i in range(x):
        v *= i
    return v


@bigO.track(lambda xs: len(xs))
def factorialize(xs: list[int]) -> list[int]:
    new_list = [fact(x) for x in xs]
    return new_list


# Exercise the same input
for i in range(30):
    factorialize([i for i in range(i * 20)])
