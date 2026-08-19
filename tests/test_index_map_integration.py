from strideweave.core.layout import Shape
from strideweave.core.permutation import Permutation
from strideweave.core.product import Product


def test_product_packs_real_permutation_children():
    first = Permutation([4, 3], 10)
    second = Permutation([2, 9], 15)

    product = Product(first, second)

    assert product.shape == Shape(2, 2)
    assert product.target_shape == Shape(10, 15)
    assert (first(0), second(0)) == (4, 2)
    assert product((0, 0)) == 24
    assert (first(1), second(1)) == (3, 9)
    assert product((1, 1)) == 93
