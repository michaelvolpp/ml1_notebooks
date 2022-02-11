import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

COMPONENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
MARKER_SIZE = 20


def initialize_gmm(n_components: int, d_x: int):
    log_weights = np.log(1 / n_components * np.ones((n_components,)))
    means = np.random.normal(size=(n_components, d_x))
    covariances = np.repeat(np.eye(d_x)[None, :, :], repeats=n_components, axis=0)

    # check output
    assert log_weights.shape == (n_components,)
    assert means.shape == (n_components, d_x)
    assert covariances.shape == (n_components, d_x, d_x)
    return log_weights, means, covariances


def compute_log_component_densities_gmm(
    X: np.array, means: np.array, covariances: np.array
):
    # check input
    n_data = X.shape[0]
    n_components = means.shape[0]
    d_x = X.shape[1]
    assert X.shape == (n_data, d_x)
    assert means.shape == (n_components, d_x)
    assert covariances.shape == (n_components, d_x, d_x)

    # compute log component densities
    log_component_densities = np.zeros((n_data, n_components))
    for k in range(n_components):
        log_component_densities[:, k] = multivariate_normal.logpdf(
            x=X, mean=means[k], cov=covariances[k]
        )

    # check output
    assert log_component_densities.shape == (n_data, n_components)
    return log_component_densities


def compute_log_joint_densities_gmm(
    X: np.array, log_weights: np.array, means: np.array, covariances: np.array
):
    # check input
    n_data = X.shape[0]
    n_components = log_weights.shape[0]
    d_x = X.shape[1]
    assert X.shape == (n_data, d_x)
    assert log_weights.shape == (n_components,)
    assert means.shape == (n_components, d_x)
    assert covariances.shape == (n_components, d_x, d_x)

    # compute log joint densities
    log_component_densities = compute_log_component_densities_gmm(
        X=X, means=means, covariances=covariances
    )
    assert log_component_densities.shape == (n_data, n_components)
    log_joint_densities = log_weights[None, :] + log_component_densities

    # check output
    assert log_joint_densities.shape == (n_data, n_components)
    return log_joint_densities


def compute_log_marginal_likelihood_gmm(
    X: np.ndarray, log_weights: np.ndarray, means: np.ndarray, covariances: np.ndarray
):
    # check input
    n_data = X.shape[0]
    n_components = log_weights.shape[0]
    d_x = X.shape[1]
    assert X.shape == (n_data, d_x)
    assert log_weights.shape == (n_components,)
    assert means.shape == (n_components, d_x)
    assert covariances.shape == (n_components, d_x, d_x)

    # compute log marginal likelihood
    log_joint_densities = compute_log_joint_densities_gmm(
        X=X, log_weights=log_weights, means=means, covariances=covariances
    )
    assert log_joint_densities.shape == (n_data, n_components)
    log_marginal_likelihood = logsumexp(log_joint_densities, axis=1)
    assert log_marginal_likelihood.shape == (n_data,)
    log_marginal_likelihood = np.sum(log_marginal_likelihood)

    return log_marginal_likelihood


def compute_elbo_gmm(
    X: np.ndarray,
    log_weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    log_responsibilities: np.ndarray,
):
    # check input
    n_data = X.shape[0]
    n_components = log_weights.shape[0]
    d_x = X.shape[1]
    assert X.shape == (n_data, d_x)
    assert log_weights.shape == (n_components,)
    assert means.shape == (n_components, d_x)
    assert covariances.shape == (n_components, d_x, d_x)
    assert log_responsibilities.shape == (n_data, n_components)

    # compute log marginal likelihood
    log_joint_densities = compute_log_joint_densities_gmm(
        X=X, log_weights=log_weights, means=means, covariances=covariances
    )
    assert log_joint_densities.shape == (n_data, n_components)
    elbo = log_joint_densities - log_responsibilities
    assert elbo.shape == (n_data, n_components)
    elbo = np.exp(log_responsibilities) * elbo
    assert elbo.shape == (n_data, n_components)
    elbo = np.sum(elbo, axis=1)
    assert elbo.shape == (n_data,)
    elbo = np.sum(elbo, axis=0)

    return elbo


def compute_kl_term_gmm(
    X: np.ndarray,
    log_weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    log_responsibilities: np.ndarray,
):
    # check input
    n_data = X.shape[0]
    n_components = log_weights.shape[0]
    d_x = X.shape[1]
    assert X.shape == (n_data, d_x)
    assert log_weights.shape == (n_components,)
    assert means.shape == (n_components, d_x)
    assert covariances.shape == (n_components, d_x, d_x)
    assert log_responsibilities.shape == (n_data, n_components)

    # compute KL term
    # we can re-use the e-step to compute the true posterior
    log_posterior = e_step_gmm(
        X=X, log_weights=log_weights, means=means, covariances=covariances
    )
    assert log_posterior.shape == (n_data, n_components)
    kl_term = log_responsibilities - log_posterior
    assert kl_term.shape == (n_data, n_components)
    kl_term = np.exp(log_responsibilities) * kl_term
    assert kl_term.shape == (n_data, n_components)
    kl_term = np.sum(kl_term, axis=1)
    assert kl_term.shape == (n_data,)
    kl_term = np.sum(kl_term, axis=0)

    # check output
    return kl_term


def e_step_gmm(
    X: np.array, log_weights: np.array, means: np.array, covariances: np.array
):
    # check input
    n_data = X.shape[0]
    n_components = log_weights.shape[0]
    d_x = X.shape[1]
    assert X.shape == (n_data, d_x)
    assert log_weights.shape == (n_components,)
    assert means.shape == (n_components, d_x)
    assert covariances.shape == (n_components, d_x, d_x)

    # compute responsibilities
    log_joint_densities = compute_log_joint_densities_gmm(
        X=X, log_weights=log_weights, means=means, covariances=covariances
    )
    assert log_joint_densities.shape == (n_data, n_components)
    log_normalizer = logsumexp(log_joint_densities, axis=1)
    assert log_normalizer.shape == (n_data,)
    log_responsibilities = log_joint_densities - log_normalizer[:, None]

    # check output
    assert log_responsibilities.shape == (n_data, n_components)
    return log_responsibilities


def m_step_gmm(
    X: np.array,
    log_responsibilities: np.array,
):
    # check input
    n_data = X.shape[0]
    d_x = X.shape[1]
    n_components = log_responsibilities.shape[1]
    assert X.shape == (n_data, d_x)
    assert log_responsibilities.shape == (n_data, n_components)

    ## update parameters
    # log weights
    n_data_k = np.exp(logsumexp(log_responsibilities, axis=0))
    assert n_data_k.shape == (n_components,)
    new_log_weights = np.log(n_data_k) - np.log(n_data)
    # means
    responsibilities = np.exp(log_responsibilities)
    new_means = responsibilities[:, :, None] * X[:, None, :]
    assert new_means.shape == (n_data, n_components, d_x)
    new_means = 1 / n_data_k[:, None] * np.sum(new_means, axis=0)
    assert new_means.shape == (n_components, d_x)
    # covariances
    diff = X[:, None, :] - new_means[None]
    assert diff.shape == (n_data, n_components, d_x)
    new_covariances = np.einsum("...i,...j->...ij", diff, diff)
    assert new_covariances.shape == (n_data, n_components, d_x, d_x)
    new_covariances = responsibilities[:, :, None, None] * new_covariances
    assert new_covariances.shape == (n_data, n_components, d_x, d_x)
    new_covariances = 1 / n_data_k[:, None, None] * np.sum(new_covariances, axis=0)

    # check outputs
    assert new_log_weights.shape == (n_components,)
    assert new_means.shape == (n_components, d_x)
    assert new_covariances.shape == (n_components, d_x, d_x)
    return new_log_weights, new_means, new_covariances


def visualize_em_gmm(
    X,
    log_weights=None,
    means=None,
    covariances=None,
    log_responsibilities=None,
):
    def plot_gaussian_ellipse(ax, mean, covariance, color):
        n_plot = 100
        evals, evecs = np.linalg.eig(covariance)
        theta = np.linspace(0, 2 * np.pi, n_plot)
        ellipsis = (np.sqrt(evals[None, :]) * evecs) @ [np.sin(theta), np.cos(theta)]
        ellipsis = ellipsis + mean[:, None]
        ax.plot(ellipsis[0, :], ellipsis[1, :], color=color, linewidth=4)

    # check input
    n_data = X.shape[0]
    d_x = X.shape[1]
    assert X.shape == (n_data, d_x)
    if log_weights is not None:
        n_components = log_weights.shape[0]
        assert log_weights.shape == (n_components,)
        assert means.shape == (n_components, d_x)
        assert covariances.shape == (n_components, d_x, d_x)
    else:
        n_components = 0
    if log_responsibilities is not None:
        assert log_responsibilities.shape == (n_data, n_components)

    # generate plot
    n_cols = 1 if log_weights is None else 2
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, 6), squeeze=False)

    # plot data
    ax = axes[0, 0]
    if log_responsibilities is None:
        ax.scatter(X[:, 0], X[:, 1], color="0.8", s=MARKER_SIZE)
    else:
        for k in range(n_components):
            alpha = np.exp(log_responsibilities[:, k])
            color = COMPONENT_COLORS[k % len(COMPONENT_COLORS)]
            ax.scatter(X[:, 0], X[:, 1], alpha=alpha, color=color, s=MARKER_SIZE)

    # plot GMM
    if log_weights is not None:
        colors = []  # log the colors to plot the weight distribution
        for k in range(n_components):
            color = COMPONENT_COLORS[k % len(COMPONENT_COLORS)]
            colors.append(color)
            plot_gaussian_ellipse(
                ax=ax, mean=means[k], covariance=covariances[k], color=color
            )

    # set title
    title = "Data"
    if log_responsibilities is not None:
        title += " + Responsibilities"
    if log_weights is not None:
        title += " + GMM"
    ax.set_title(title)
    ax.axis("scaled")

    # plot weights
    if log_weights is not None:
        ax = axes[0, 1]
        weights = np.exp(log_weights)
        ax.pie(weights, labels=[f"{w*100:.2f}%" for w in weights], colors=colors)
        ax.axis("scaled")
        ax.set_title("Mixture weights")

    plt.show()


def visualize_log_marginal_likelihood_decomposition(ax, elbo, kl_term):
    pass
