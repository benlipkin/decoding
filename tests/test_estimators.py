import re
import time

import jax.numpy as jnp

from decoding.estimators import MAP, MBR, SelfConsistency, commutativeMBR, linearMBR
from decoding.metrics import levenshtein
from decoding.pmf import CategoricalLogPMF


def test_MBR() -> None:
    def u1(s1: str, s2: str) -> int:
        return int(s1 < s2)

    c = "cba"
    d = CategoricalLogPMF.from_samples(c)
    assert [s.item for s in MBR(d, utility=u1)] == sorted(c)

    def u2(_1: str, _2: str) -> float:
        return 1.0

    logits = jnp.asarray([1.0, 2.0, 3.0])
    d = CategoricalLogPMF.from_logits(logits=logits, cats=c)
    assert [s.item for s in MBR(d, utility=u2)] == list(c)[::-1]
    assert [s.item for s in MBR(d, utility=u2, parallelize=True)] == list(c)[::-1]


def test_commutativeMBR() -> None:
    def u(s1: str, s2: str) -> int:
        time.sleep(1e-3)
        return -levenshtein(s1, s2)

    c = ["car", "can", "cat", "bat", "hat"]
    d = CategoricalLogPMF.from_samples(c)

    t1 = time.time()
    s1 = MBR(d, utility=u)
    t1 = time.time() - t1

    t2 = time.time()
    s2 = commutativeMBR(d, utility=u)
    t2 = time.time() - t2

    t3 = time.time()
    s3 = commutativeMBR(d, utility=u, parallelize=True)
    t3 = time.time() - t3

    assert s1[0].item == s2[0].item == s3[0].item == "cat"
    assert t3 < t2 < t1


def test_linearMBR() -> None:
    def u(s1: str, s2: str) -> int:
        time.sleep(1e-2)
        return len(s1) + len(s2)

    c = ["a", "aaaaaa", "aa", "aaaa", "aaaaa", "", "aaa"]
    d = CategoricalLogPMF.from_samples(c)

    t1 = time.time()
    o1 = MBR(d, utility=u)
    t1 = time.time() - t1

    t2 = time.time()
    o2 = commutativeMBR(d, utility=u)
    t2 = time.time() - t2

    t3 = time.time()
    o3 = linearMBR(d, utility=len)
    t3 = time.time() - t3

    t4 = time.time()
    o4 = linearMBR(d, utility=len, parallelize=True)
    t4 = time.time() - t4

    assert (
        [s.item for s in o1]
        == [s.item for s in o2]
        == [s.item for s in o3]
        == [s.item for s in o4]
    )
    assert o3[0].item == "aaaaaa"
    assert o3[-1].item == ""
    assert t4 < t3 < t2 < t1


def test_MAP() -> None:
    c = [1, 2, 3, 4, 5]
    logits = jnp.asarray(c).astype(float)
    d = CategoricalLogPMF.from_logits(logits=logits, cats=c)
    assert [s.item for s in MAP(d)] == c[::-1]


def test_SelfConsistency() -> None:
    def postproc(s: str) -> str:
        m = re.match(r".*\nANSWER:\n(.*)\n\n", s)
        return m.groups()[0] if m else ""

    def filt(s: str) -> bool:
        return s != ""

    c = [
        "abc\nANSWER:\n1\n\n",
        "def\nANSWER:\n2\n\n",
        "ghi\nANSWER:\n2\n\n",
        "ill formed",
        "bad response",
        "will get filtered",
    ]
    d = CategoricalLogPMF.from_samples(c)
    assert SelfConsistency(d, postproc=postproc, filt=filt)[0].item == "2"
