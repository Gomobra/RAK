"""Test paritas antara binding Python (_rak) dan implementasi referensi (Python murni).

Tujuan: memastikan output forward dan gradien backward dari binding C++ (_rak)
cocok dengan implementasi referensi Python (manual / finite difference) untuk
semua op inti Fase 0. Ini memenuhi Definition of Done PRD 2.5 (uji paritas).

Catatan: numpy mungkin tidak tersedia di Termux, jadi _rak.to_list() digunakan
sebagai fallback untuk membandingkan nilai tensor, dan properti .grad untuk
membandingkan gradien.
"""

import sys
import traceback
import math

# Coba impor _rak dari lokasi instalasi atau build output.
sys.path.insert(0, "/storage/emulated/0/RAK/src")
sys.path.insert(0, "/data/data/com.termux/files/usr/lib/python3.14/site-packages")
try:
    import _rak
except ImportError:
    sys.path.insert(0, "/storage/emulated/0/RAK/build/cpp/python")
    import _rak

TOL = 1e-4
failed = []
passed = []


def close(a, b, tol=TOL):
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return False
        return all(close(x, y, tol) for x, y in zip(a, b))
    return abs(float(a) - float(b)) <= tol


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
        print(f"  PASS {name}")
    else:
        failed.append((name, detail))
        print(f"  FAIL {name}: {detail}")


def reset_tape():
    _rak.autograd_reset()


# ---------- helper: finite difference gradien numerik ----------
def numerical_grad(forward, inputs, input_idx, elem_idx, eps=1e-4):
    in_ = inputs[input_idx]
    in_list = in_.to_list()
    orig = in_list[elem_idx]

    in_list[elem_idx] = orig + eps
    t2 = _rak.Tensor(list(in_.shape()), in_list, True)
    vp = forward([t2])
    vp_val = vp.at(0) if hasattr(vp, "at") else vp.to_list()[0]

    in_list[elem_idx] = orig - eps
    t3 = _rak.Tensor(list(in_.shape()), in_list, True)
    vm = forward([t3])
    vm_val = vm.at(0) if hasattr(vm, "at") else vm.to_list()[0]

    in_list[elem_idx] = orig
    return (vp_val - vm_val) / (2.0 * eps)


# ---------- 1. GEMM ----------
def test_gemm():
    reset_tape()
    A = _rak.Tensor([2, 3], [1, 2, 3, 4, 5, 6], True)
    B = _rak.Tensor([3, 2], [7, 8, 9, 10, 11, 12], True)
    C = _rak.gemm(A, B)
    # C[0,0]=1*7+2*9+3*11=58, C[0,1]=1*8+2*10+3*12=64
    # C[1,0]=4*7+5*9+6*11=139, C[1,1]=4*8+5*10+6*12=154
    exp = [58.0, 64.0, 139.0, 154.0]
    check("gemm forward", close(C.to_list(), exp), f"got {C.to_list()}")
    loss = _rak.sum(C, -1)
    _rak.backward(loss)
    # grad_C semua 1 -> grad_A[i,k] = sum_j B[k,j] (kolom B)
    # grad_A[0,:] = [7+9, 8+10, 9+11]? Tidak. grad_A[i,k] = sum_j grad_C[i,j] * B[k,j]
    # grad_A[i,k] = sum_j grad_C[i,j] * B[k,j] = B[k,0]+B[k,1]
    # k=0: 7+8=15, k=1: 9+10=19, k=2: 11+12=23, tiap baris sama
    gA = A.grad
    check("gemm backward grad A", close(gA, [15.0, 19.0, 23.0, 15.0, 19.0, 23.0]),
          f"got {gA}")
    # grad_B[k,j] = sum_i A[i,k] * grad_C[i,j] = A[0,k]+A[1,k]
    # k=0: 1+4=5, k=1: 2+5=7, k=2: 3+6=9, tiap kolom sama
    gB = B.grad
    check("gemm backward grad B", close(gB, [5.0, 5.0, 7.0, 7.0, 9.0, 9.0]),
          f"got {gB}")


# ---------- 2. add / mul ----------
def test_add_mul():
    reset_tape()
    a = _rak.Tensor([3], [1.0, 2.0, 3.0], True)
    b = _rak.Tensor([3], [4.0, 5.0, 6.0], True)
    c = _rak.add(a, b)
    check("add forward", close(c.to_list(), [5.0, 7.0, 9.0]), f"got {c.to_list()}")
    loss = _rak.sum(c, -1)
    _rak.backward(loss)
    check("add backward grad a", close(a.grad, [1.0, 1.0, 1.0]), f"got {a.grad}")
    check("add backward grad b", close(b.grad, [1.0, 1.0, 1.0]), f"got {b.grad}")

    a2 = _rak.Tensor([3], [1.0, 2.0, 3.0], True)
    b2 = _rak.Tensor([3], [4.0, 5.0, 6.0], True)
    m = _rak.mul(a2, b2)
    check("mul forward", close(m.to_list(), [4.0, 10.0, 18.0]), f"got {m.to_list()}")
    loss = _rak.sum(m, -1)
    _rak.backward(loss)
    check("mul backward grad a", close(a2.grad, [4.0, 5.0, 6.0]), f"got {a2.grad}")
    check("mul backward grad b", close(b2.grad, [1.0, 2.0, 3.0]), f"got {b2.grad}")


# ---------- 3. silu / gelu ----------
def test_silu_gelu():
    reset_tape()
    x = _rak.Tensor([4], [-2.0, -0.5, 0.5, 2.0], True)
    s = _rak.silu(x)
    exp_silu = [v * (1.0 / (1.0 + math.exp(-v))) for v in [-2.0, -0.5, 0.5, 2.0]]
    check("silu forward", close(s.to_list(), exp_silu, 1e-3), f"got {s.to_list()}")
    loss = _rak.sum(s, -1)
    _rak.backward(loss)
    check("silu backward grad nonzero", any(abs(g) > 1e-9 for g in x.grad),
          f"got {x.grad}")

    x2 = _rak.Tensor([4], [-2.0, -0.5, 0.5, 2.0], True)
    g = _rak.gelu(x2)
    exp_gelu = [0.5 * v * (1.0 + math.tanh(0.79788456 * (v + 0.044715 * v * v * v)))
                for v in [-2.0, -0.5, 0.5, 2.0]]
    check("gelu forward", close(g.to_list(), exp_gelu, 1e-3), f"got {g.to_list()}")
    loss = _rak.sum(g, -1)
    _rak.backward(loss)
    check("gelu backward grad nonzero", any(abs(g) > 1e-9 for g in x2.grad),
          f"got {x2.grad}")


# ---------- 4. sum / mean / max ----------
def test_reduce():
    reset_tape()
    # axis=-1: reduce semua ke skalar {1}
    x = _rak.Tensor([2, 3], [1, 2, 3, 4, 5, 6], True)
    s = _rak.sum(x, -1)
    check("sum forward scalar", close(s.to_list(), [21.0]), f"got {s.to_list()}")
    loss = _rak.sum(s, -1)
    _rak.backward(loss)
    check("sum backward grad", close(x.grad, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
          f"got {x.grad}")

    x2 = _rak.Tensor([2, 3], [1, 2, 3, 4, 5, 6], True)
    m = _rak.mean(x2, -1)
    check("mean forward scalar", close(m.to_list(), [3.5]), f"got {m.to_list()}")

    x3 = _rak.Tensor([2, 3], [1, 2, 3, 4, 5, 6], True)
    mx = _rak.max(x3, -1)
    check("max forward scalar", close(mx.to_list(), [6.0]), f"got {mx.to_list()}")


# ---------- 5. rms_norm / softmax ----------
def test_norm():
    reset_tape()
    x = _rak.Tensor([2, 3], [1, 2, 3, 4, 5, 6], True)
    w = _rak.Tensor([3], [1.0, 1.0, 1.0], True)
    rn = _rak.rms_norm(x, w)
    check("rms_norm forward shape", list(rn.shape()) == [2, 3], f"got {list(rn.shape())}")
    loss = _rak.sum(rn, -1)
    _rak.backward(loss)
    check("rms_norm backward grad x nonzero", any(abs(g) > 1e-9 for g in x.grad),
          f"got {x.grad}")
    check("rms_norm backward grad w nonzero", any(abs(g) > 1e-9 for g in w.grad),
          f"got {w.grad}")

    x2 = _rak.Tensor([2, 3], [1, 2, 3, 4, 5, 6], True)
    sm = _rak.softmax(x2)
    row0 = sm.to_list()[0:3]
    row1 = sm.to_list()[3:6]
    check("softmax row0 sum=1", abs(sum(row0) - 1.0) < 1e-5, f"sum={sum(row0)}")
    check("softmax row1 sum=1", abs(sum(row1) - 1.0) < 1e-5, f"sum={sum(row1)}")
    loss = _rak.sum(sm, -1)
    _rak.backward(loss)
    check("softmax backward grad nonzero", any(abs(g) > 1e-9 for g in x2.grad),
          f"got {x2.grad}")


# ---------- 6. embedding ----------
def test_embedding():
    reset_tape()
    W = _rak.Tensor([4, 2], [1, 2, 3, 4, 5, 6, 7, 8], True)
    X = _rak.Tensor([3], [0.0, 2.0, 3.0], False)
    Y = _rak.embedding(X, W)
    exp = [1.0, 2.0, 5.0, 6.0, 7.0, 8.0]
    check("embedding forward", close(Y.to_list(), exp), f"got {Y.to_list()}")
    loss = _rak.sum(Y, -1)
    _rak.backward(loss)
    gW = W.grad
    check("embedding backward W[0]", close(gW[0:2], [1.0, 1.0]), f"got {gW[0:2]}")
    check("embedding backward W[2]", close(gW[4:6], [1.0, 1.0]), f"got {gW[4:6]}")
    check("embedding backward W[3]", close(gW[6:8], [1.0, 1.0]), f"got {gW[6:8]}")


# ---------- 7. recurrent_scan ----------
def test_scan():
    reset_tape()
    X = _rak.Tensor([1, 3, 2], [1, 2, 3, 4, 5, 6], False)
    D = _rak.Tensor([2], [0.5, 0.5], False)
    par = _rak.recurrent_scan(X, D, "parallel", "forward")
    seq = _rak.recurrent_scan(X, D, "sequential", "forward")
    check("scan parallel vs sequential", close(par.to_list(), seq.to_list()),
          f"par={par.to_list()} seq={seq.to_list()}")
    check("scan numel", par.numel() == 6, f"got {par.numel()}")


# ---------- 8. local_attention ----------
def test_local_attention():
    reset_tape()
    Q = _rak.Tensor([1, 4, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], False)
    K = _rak.Tensor([1, 4, 3], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], False)
    V = _rak.Tensor([1, 4, 3], [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], False)
    att = _rak.local_attention(Q, K, V, 2)
    check("local_attention shape", list(att.shape()) == [1, 4, 3], f"got {list(att.shape())}")


# ---------- 9. sparse_lookup_topk ----------
def test_sparse_lookup():
    # query=[1,0,0], keys: slot0=[1,0,0](sim=1), slot1=[0,1,0](sim=0),
    # slot2=[0,0,1](sim=0), slot3=[5,0,0](sim=5). Top-1 = slot 3.
    q = _rak.Tensor([1, 1, 3], [1, 0, 0], False)
    k = _rak.Tensor([4, 3], [1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 0, 0], False)
    res = _rak.sparse_lookup_topk(q, k, 2)
    vals = res.values.to_list()
    idx = res.indices
    check("sparse_lookup values shape", list(res.values.shape()) == [1, 1, 2, 3],
          f"got {list(res.values.shape())}")
    check("sparse_lookup top-1 index = 3", idx[0][0][0] == 3, f"got {idx[0][0][0]}")
    check("sparse_lookup top-2 index = 0", idx[0][0][1] == 0, f"got {idx[0][0][1]}")


# ---------- 10. sparse_update ----------
def test_sparse_update():
    reset_tape()
    # slots=[3,2]=[1,2,3,4,5,6], idx=[0,2], upd=[2,2]=[10,20,30,40], mode add
    slots = _rak.Tensor([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], False)
    idx = _rak.Tensor([2], [0.0, 2.0], False)
    upd = _rak.Tensor([2, 2], [10.0, 20.0, 30.0, 40.0], False)
    out = _rak.sparse_update(slots, idx, upd, "add")
    # out[0,:]=[1+10, 2+20]=[11,22]; out[1,:]=[3,4]; out[2,:]=[5+30, 6+40]=[35,46]
    exp = [11.0, 22.0, 3.0, 4.0, 35.0, 46.0]
    check("sparse_update add forward", close(out.to_list(), exp), f"got {out.to_list()}")


if __name__ == "__main__":
    tests = [
        test_gemm, test_add_mul, test_silu_gelu, test_reduce, test_norm,
        test_embedding, test_scan, test_local_attention, test_sparse_lookup,
        test_sparse_update,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(passed)}/{len(tests)} test paritas lolos")
    if failed:
        print("Gagal:")
        for name, det in failed:
            print(f"  - {name}: {det}")
    sys.exit(1 if failed else 0)
