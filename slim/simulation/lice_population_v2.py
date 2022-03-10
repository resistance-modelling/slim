from numba import njit

from .lice_population import GenoDistrib, from_ratios
import numpy as np

# TODO: move that into a test suite.

if __name__ == "__main__":

    """
    test = GenoDistrib(2, np.array([0.2, 0.8, 0.2]))
    test["a"] = 50
    test["A"] = 50
    test["Aa"] = 100
    print(test["A"])
    print(test["b"])
    test["b"] = 100
    print(test.to_json_dict())
    test["B"] = 50
    test["Bb"] = 50
    print(test["A"])
    print(test["b"])

    print(test.to_json_dict())
    test2 = test.normalise_to(1000)
    print(test2.gross)
    print(test2._store)
    print(test.normalise_to(0))
    print(test.add(test2)._store)
    print(test.sub(test2)._store)
    print(test.mul(test2)._store)
    print(test.mul_by_scalar(1 / 3)._store)
    test._truncate_negatives()
    """

    @njit
    def test_from_ratios():
        test3 = from_ratios({"A": 0.1, "Aa": 0.2, "a": 0.3}).normalise_to(10)
        print(test3.to_json_dict())

    test_from_ratios()
