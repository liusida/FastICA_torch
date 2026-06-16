import torch

from fastica_torch import FastICA, _positive_cube


def test_positive_cube_is_oriented():
    x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    gx_pos, g_x_pos = _positive_cube(x)
    gx_neg, g_x_neg = _positive_cube(-x)

    assert torch.all(gx_pos[:, :3] == 0)
    assert torch.all(gx_pos[:, 3:] > 0)
    assert not torch.allclose(gx_pos, -gx_neg)
    assert torch.all(g_x_pos >= 0)
    assert torch.all(g_x_neg >= 0)


def test_positive_cube_fits():
    X = torch.randn(300, 6)

    ica = FastICA(
        n_components=3,
        fun="positive_cube",
        random_state=42,
        max_iter=100,
    )
    S = ica.fit_transform(X)

    assert S.shape == (300, 3)
    assert torch.all(torch.isfinite(S))
    assert ica.components_.shape == (3, 6)
