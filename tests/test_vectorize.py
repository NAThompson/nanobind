import pytest
import test_vectorize_ext as t

np = pytest.importorskip("numpy")


def test_one_arg():
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert np.array_equal(t.square(x), x * x)
    assert t.square(3.0).shape == ()
    assert t.square(3.0) == 9.0


def test_two_args_and_conversion():
    a, b = np.arange(4.0), np.arange(4.0) + 1
    assert np.array_equal(t.add(a, b), a + b)
    assert np.array_equal(t.add(np.arange(3, dtype=np.float32), np.arange(3)), 2 * np.arange(3))
    # Implicit dtype conversion, mixed argument types
    assert np.array_equal(t.mul3(np.arange(3), 2, np.int32(5)), np.arange(3) * 10)


def test_broadcast():
    col, row = np.arange(3.0)[:, None], np.arange(4.0)
    assert np.array_equal(t.add(col, row), col + row)
    assert np.array_equal(t.hyp(3.0, np.array([4.0, 0.0])), [5.0, 3.0])
    assert t.add(np.ones((2, 1, 3)), np.ones((4, 1))).shape == (2, 4, 3)
    assert t.add(np.empty((0, 3)), 1.0).shape == (0, 3)


def test_strided():
    a = np.arange(12.0).reshape(3, 4)
    assert np.array_equal(t.add(a.T, 1.0), a.T + 1)
    assert np.array_equal(t.add(a[::2, ::-2], a[1, :2]), a[::2, ::-2] + a[1, :2])


def test_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        t.add(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="shape mismatch"):
        t.add(np.ones((2, 3)), np.ones((3, 2)))
