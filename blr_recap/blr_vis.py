import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from blr import BayesianLinearRegression


def create_w_grid(n_w):
    w0 = np.linspace(-1.0, 1.0, n_w)
    w1 = np.linspace(-1.0, 1.0, n_w)
    ww0, ww1 = np.meshgrid(w0, w1)
    ws = np.vstack([ww0.ravel(), ww1.ravel()]).T
    return ww0, ww1, ws


def plot_contour(
    ww0: np.ndarray,
    ww1: np.ndarray,
    vals: np.ndarray,
    w_true: np.ndarray,
):
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(5, 5))
    ax = axes[0, 0]
    h = ax.contourf(ww0, ww1, vals, levels=250, cmap="plasma", vmax=0.0, vmin=-5.0)
    ax.scatter(w_true[0], w_true[1], s=20, c="r", marker="x")
    ax.set_xlabel("$w_0$")
    ax.set_ylabel("$w_1$")
    ax.axis("scaled")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    fig.tight_layout()


def plot_hypotheses(
    model: BayesianLinearRegression,
    ws: np.ndarray,
    vals: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w_true: np.ndarray,
):
    n_x = x.shape[0]
    n_w = int(np.sqrt(ws.shape[0]))
    fig, axes = plt.subplots(
        nrows=n_w, ncols=n_w, sharex=True, sharey=True, squeeze=False, figsize=(5, 5)
    )
    x_plot = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y_true = model.predict(x_plot, w_true.reshape(-1, 2))[0]

    # filter out values below max(vals) - 8
    # idx = vals > vals.max() - (n_x + 1) * 4
    idx = vals > -np.inf
    lws = vals - vals.min()
    lws = lws / lws.max()
    lws = np.exp(20 * (lws - 1)) * 3
    for i, (lw, w) in enumerate(zip(lws, ws)):
        ax = axes[i // n_w, i % n_w]
        if idx[i]:
            y_plot = model.predict(x_plot, w.reshape(-1, 2))[0]
            ax.plot(x_plot, y_plot, linewidth=lw, color="k")
        ax.plot(x_plot, y_true, alpha=0.1, color="r")
        ax.scatter(x, y, marker="x", s=1, color="r")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
    fig.tight_layout()


def visualize_log_likelihood(
    model: BayesianLinearRegression,
    x: np.ndarray,
    y: np.ndarray,
    w_true: np.ndarray,
    plot_type: str,
):
    assert plot_type in ["contour", "hypotheses"]
    assert x.ndim == y.ndim == 2
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] >= 1
    assert y.shape[1] == y.shape[1] == 1

    # create w-grid
    n_w = 20 if plot_type == "contour" else 7
    ww0, ww1, ws = create_w_grid(n_w=n_w)

    # compute function values
    vals = model.log_likelihood(w=ws, x=x, y=y)
    # vals = vals / x.shape[0]

    # plot log likelihood over w
    if plot_type == "hypotheses":
        plot_hypotheses(
            model=model,
            ws=ws,
            vals=vals,
            x=x,
            y=y,
            w_true=w_true,
        )
    elif plot_type == "contour":
        vals = vals.reshape(n_w, n_w)
        plot_contour(
            ww0=ww0,
            ww1=ww1,
            vals=vals,
            w_true=w_true,
        )

    plt.show()


def visualize_log_posterior(
    model: BayesianLinearRegression,
    x: np.ndarray,
    y: np.ndarray,
    w_true: np.ndarray,
    plot_type: str,
):
    assert plot_type in ["contour", "hypotheses"]
    assert x.ndim == y.ndim == 2
    assert x.shape[0] == y.shape[0]
    assert y.shape[1] == y.shape[1] == 1

    # create w-grid
    n_w = 20 if plot_type == "contour" else 7
    ww0, ww1, ws = create_w_grid(n_w)

    # compute function values
    vals = model.log_posterior_pdf(w=ws, x=x, y=y)
    # vals = vals / (x.shape[0] + 1)

    # plot log likelihood over w
    if plot_type == "hypotheses":
        plot_hypotheses(
            model=model,
            ws=ws,
            vals=vals,
            x=x,
            y=y,
            w_true=w_true,
        )
    elif plot_type == "contour":
        vals = vals.reshape(n_w, n_w)
        plot_contour(
            ww0=ww0,
            ww1=ww1,
            vals=vals,
            w_true=w_true,
        )

    plt.show()


def visualize_predictive_distribution(
    model: BayesianLinearRegression,
    x: np.ndarray,
    y: np.ndarray,
    w_true: np.ndarray,
):
    # obtain ground truth, samples, and predictive distribution
    x_plot = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y_true = model.predict(x_plot, w_true.reshape(-1, 2))[0]
    y_samples = model.sample_function(x_q=x_plot, x=x, y=y, S=11)
    mu_pred, var_pred = model.predictive_distribution(x_q=x_plot, x=x, y=y)
    sigma_pred = np.sqrt(var_pred)

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # ground truth
    ax.plot(x_plot, y_true, color="r", label="ground truth")
    ax.scatter(x, y, color="r", marker="x", s=50)
    # samples
    for i in range(y_samples.shape[0]):
        ax.plot(x_plot, y_samples[i], "b", ls="--", alpha=0.25)
    # predictive distribution
    ax.plot(x_plot, mu_pred, "b", label="predictive distr.")
    ax.fill_between(
        x_plot.squeeze(),
        mu_pred - 2 * sigma_pred,
        mu_pred + 2 * sigma_pred,
        color="b",
        alpha=0.25,
    )
    ax.set_title("BLR Predictive Distribution")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()
