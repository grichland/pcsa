"""The numpy sprite network: its convolution, its output, its loader's checks, and (torch) its parity."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from pcsa.common.errors import ConfigError
from pcsa.phone import sprite_net


def random_weights(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: (rng.standard_normal(s) * 0.2).astype(np.float32) for k, s in sprite_net.weight_shapes().items()}


def naive_conv(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, c, h, wd = x.shape
    p = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((n, len(w), h, wd), np.float64)
    for o in range(len(w)):
        for y in range(h):
            for xx in range(wd):
                out[:, o, y, xx] = (p[:, :, y : y + 3, xx : xx + 3] * w[o]).sum(axis=(1, 2, 3)) + b[o]
    return out


def test_the_convolution_is_a_padded_3x3_cross_correlation() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 6, 5)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    assert np.allclose(sprite_net._conv3x3(x, w, b), naive_conv(x, w, b), atol=1e-4)


def test_forward_gives_one_unit_vector_per_input() -> None:
    rng = np.random.default_rng(2)
    masks = (rng.random((3, sprite_net.SIZE, sprite_net.SIZE)) > 0.5).astype(np.float32)
    labs = rng.integers(0, 256, (3, sprite_net.SIZE, sprite_net.SIZE, 3)).astype(np.float32)
    out = sprite_net.forward(random_weights(), *sprite_net.encode(masks, labs, np.array([0.8, 1.0, 1.3])))
    assert out.shape == (3, sprite_net.EMBED)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_encoding_zeroes_colour_outside_the_silhouette() -> None:
    masks = np.zeros((1, sprite_net.SIZE, sprite_net.SIZE), np.float32)
    labs = np.full((1, sprite_net.SIZE, sprite_net.SIZE, 3), 200, np.float32)
    x, _ = sprite_net.encode(masks, labs, np.array([1.0]))
    assert not x.any()


def write_network(folder, weights) -> None:
    def save(name: str, **arrays) -> dict:
        np.savez(folder / name, **arrays)
        return {"file": name, "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()}

    refs = {
        domain: save(
            f"{domain}_refs.npz",
            names=np.array(["Venusaur", "Venusaur-Mega"]),
            base_names=np.array(["Venusaur", "Venusaur"]),
            vectors=np.eye(2, sprite_net.EMBED, dtype=np.float32),
        )
        for domain in sprite_net.DOMAINS
    }
    metadata = {
        "schema": sprite_net.SCHEMA,
        "weights": save("sprite_net.npz", **weights),
        "references": refs,
        "log_thresholds": {"min_score": 0.7, "min_margin": 0.1},
    }
    (folder / "sprite_net.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_load_reads_weights_references_and_thresholds(tmp_path) -> None:
    write_network(tmp_path, random_weights())
    net = sprite_net.load(tmp_path / "sprite_net.json")
    assert net.references["log"].base_names["Venusaur-Mega"] == "Venusaur"
    assert (net.log_min_score, net.log_min_margin) == (0.7, 0.1)


def test_load_refuses_a_file_that_does_not_match_its_hash(tmp_path) -> None:
    write_network(tmp_path, random_weights())
    np.savez(tmp_path / "log_refs.npz", names=np.array(["X"]), base_names=np.array(["X"]), vectors=np.zeros((1, 64)))
    with pytest.raises(ConfigError, match="hash"):
        sprite_net.load(tmp_path / "sprite_net.json")


def test_load_refuses_weights_of_another_shape(tmp_path) -> None:
    weights = random_weights()
    weights["f2.w"] = weights["f2.w"][:10]
    write_network(tmp_path, weights)
    with pytest.raises(ConfigError, match="architecture"):
        sprite_net.load(tmp_path / "sprite_net.json")


@pytest.mark.torch
def test_the_exported_network_matches_torch() -> None:
    import torch

    from pcsa.sprite_training.sprite_train import Net

    torch.manual_seed(0)
    net = Net()
    for norm in net.norms:  # non-trivial running statistics, so the folding is exercised
        norm.running_mean.uniform_(-0.5, 0.5)
        norm.running_var.uniform_(0.5, 2.0)
        norm.weight.data.uniform_(0.5, 1.5)
        norm.bias.data.uniform_(-0.2, 0.2)
    net.eval()
    x = torch.rand(5, 4, sprite_net.SIZE, sprite_net.SIZE)
    a = torch.randn(5) * 0.3
    with torch.no_grad():
        theirs = net(x, a).numpy()
    ours = sprite_net.forward(net.export(), x.numpy(), a.numpy())
    assert np.abs(theirs - ours).max() < 1e-4
