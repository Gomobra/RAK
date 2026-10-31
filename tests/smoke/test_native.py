#!/usr/bin/env python3
"""Test smoke untuk modul native _rak dan paket Python rak."""
import sys
import traceback

# Coba impor _rak dari lokasi instalasi (site-packages) atau build output.
sys.path.insert(0, "/data/data/com.termux/files/usr/lib/python3.14/site-packages")
sys.path.insert(0, "/storage/emulated/0/RAK/src")
try:
    import _rak
except ImportError:
    sys.path.insert(0, "build/cpp/python")
    import _rak

def test_tensor_basic():
    import _rak
    t = _rak.zeros([2, 3], True)
    assert t.numel() == 6
    assert list(t.shape()) == [2, 3]
    assert t.ndim() == 2
    print("  PASS test_tensor_basic")
def test_add_backward():
    import _rak
    x = _rak.zeros([3], True)
    w = _rak.zeros([3], True)
    loss = _rak.sum(_rak.mul(x, w), -1)
    _rak.backward(loss)
    # grad x = w = [0,0,0], grad w = x = [0,0,0]
    gx = x.to_list()
    gw = w.to_list()
    assert gx == [0.0, 0.0, 0.0], gx
    assert gw == [0.0, 0.0, 0.0], gw
    print("  PASS test_add_backward")

def test_silu():
    import _rak
    s = _rak.silu(_rak.zeros([4], True))
    loss = _rak.sum(s, -1)
    _rak.backward(loss, retain_graph=True)
    print("  PASS test_silu")

def test_rms_norm():
    import _rak
    rn = _rak.rms_norm(_rak.zeros([2, 3], True), _rak.zeros([3], True))
    print("  PASS test_rms_norm")

def test_softmax():
    import _rak
    sm = _rak.softmax(_rak.zeros([2, 3], True))
    print("  PASS test_softmax")

def test_scan():
    import _rak
    X = _rak.zeros([1, 3, 2], False)
    D = _rak.zeros([2], False)
    sc = _rak.recurrent_scan(X, D, "parallel", "forward")
    assert sc.numel() == 6
    print("  PASS test_scan")

def test_local_attention():
    import _rak
    Q = _rak.zeros([1, 4, 3], False)
    K = _rak.zeros([1, 4, 3], False)
    V = _rak.zeros([1, 4, 3], False)
    att = _rak.local_attention(Q, K, V, 2)
    assert list(att.shape()) == [1, 4, 3]
    print("  PASS test_local_attention")

def test_sparse_lookup():
    import _rak
    q = _rak.zeros([1, 1, 3], False)
    k = _rak.zeros([4, 3], False)
    res = _rak.sparse_lookup_topk(q, k, 2)
    assert len(res.indices[0][0]) == 2
    print("  PASS test_sparse_lookup")

def test_rak_package():
    import rak
    assert hasattr(rak, "native")
    print("  PASS test_rak_package")

if __name__ == "__main__":
    tests = [
        test_tensor_basic,
        test_add_backward,
        test_silu,
        test_rms_norm,
        test_softmax,
        test_scan,
        test_local_attention,
        test_sparse_lookup,
        test_rak_package,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} test lolos")
    sys.exit(1 if failed else 0)
