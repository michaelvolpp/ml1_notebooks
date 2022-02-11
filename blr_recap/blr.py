import numpy as np
from scipy.stats import multivariate_normal, norm


class BayesianLinearRegression:
    def __init__(self, sigma_w: float, sigma_y: float, d_w: int):
        self.sigma_w = sigma_w
        self.lam = 1 / sigma_w ** 2
        self.sigma_y = sigma_y
        self.d_x = 1
        self.d_w = d_w

    def poly_feature_matrix(self, x: np.ndarray):
        # check input
        assert x.ndim == 2
        assert x.shape[1] == 1  # only works for d_x == 1

        # compute feature matrix
        Phi = np.vander(x=x.squeeze(-1), N=self.d_w, increasing=True)

        # check output
        assert Phi.shape == (x.shape[0], self.d_w)
        return Phi

    def prior_parameters(self):
        # compute prior parameters
        mean = np.zeros((self.d_w,))
        std = self.sigma_w * np.ones((self.d_w,))

        # check output
        assert mean.shape == (self.d_w,)
        assert std.shape == (self.d_w,)
        return mean, std

    def log_prior_pdf(self, w: np.ndarray):
        # check input
        assert w.ndim == 2
        assert w.shape[1] == self.d_w

        mean, scale = self.prior_parameters()
        log_prior_pdf = norm(loc=mean, scale=scale).logpdf(w).sum(axis=1)

        # check output
        assert log_prior_pdf.shape == (w.shape[0],)
        return log_prior_pdf

    def predict(self, x_q: np.ndarray, w: np.ndarray):
        # check input
        assert x_q.ndim == 2
        assert x_q.shape[1] == self.d_x
        assert w.ndim == 2
        assert w.shape[1] == self.d_w

        # compute predictions
        f_x = w @ self.poly_feature_matrix(x_q).T

        # check output
        assert f_x.shape == (w.shape[0], x_q.shape[0])
        return f_x

    def sample_function(self, x_q: np.ndarray, x: np.ndarray, y: np.ndarray, S: int):
        # check input
        assert x.ndim == y.ndim == x_q.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == x_q.shape[1] == self.d_x
        assert y.shape[1] == 1

        # sample weights from posterior
        mean, cov = self.posterior_parameters(x=x, y=y)
        ws = multivariate_normal(mean=mean, cov=cov).rvs(size=S)
        # compute corresponding functions
        f_x = self.predict(x_q=x_q, w=ws)

        # check output
        assert f_x.shape == (S, x_q.shape[0])
        return f_x

    def log_likelihood(self, w: np.ndarray, x: np.ndarray, y: np.ndarray):
        # check input
        assert x.ndim == y.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.d_x
        assert y.shape[1] == 1
        assert w.ndim == 2
        assert w.shape[1] == self.d_w

        # compute log likelihood
        mean, scale = self.predict(x_q=x, w=w), self.sigma_y
        log_lhd = norm(loc=mean, scale=scale).logpdf(y.T).sum(axis=1)

        # check output
        assert log_lhd.shape == (w.shape[0],)
        return log_lhd

    def posterior_parameters(self, x: np.ndarray, y: np.ndarray):
        # check input
        assert x.ndim == y.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.d_x
        assert y.shape[1] == 1

        # compute posterior parameters
        Phi = self.poly_feature_matrix(x)
        A = Phi.T @ Phi + self.lam * self.sigma_y ** 2 * np.eye(self.d_w)
        mat_mean = np.linalg.solve(A, Phi.T @ y)  # use numerically stable operation
        mat_cov = np.linalg.inv(A)  # we have to bite the bullet and compute the inverse
        mean = mat_mean[:, 0]
        cov = self.sigma_y ** 2 * mat_cov

        # check output
        assert mean.shape == (self.d_w,)
        assert cov.shape == (self.d_w, self.d_w)
        return mean, cov

    def log_posterior_pdf(self, w: np.ndarray, x: np.ndarray, y: np.ndarray):
        # check input
        assert x.ndim == y.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.d_x
        assert y.shape[1] == 1
        assert w.ndim == 2
        assert w.shape[1] == self.d_w

        # check input
        mean, cov = self.posterior_parameters(x=x, y=y)
        log_post_pdf = multivariate_normal(mean=mean, cov=cov).logpdf(w)

        # check output
        assert log_post_pdf.shape == (w.shape[0],)
        return log_post_pdf

    def predictive_distribution(self, x_q: np.ndarray, x: np.ndarray, y: np.ndarray):
        # check input
        assert x.ndim == y.ndim == x_q.ndim == 2
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == x_q.shape[1] == self.d_x
        assert y.shape[1] == 1

        # compute predictive distribution
        phi = self.poly_feature_matrix(x_q)
        Phi = self.poly_feature_matrix(x)
        A = Phi.T @ Phi + self.lam * self.sigma_y ** 2 * np.eye(Phi.shape[1])
        mat_mean = np.linalg.solve(A, Phi.T @ y)
        mat_var = np.linalg.solve(A, phi.T)
        mean = (phi @ mat_mean)[:, 0]
        var = self.sigma_y ** 2 * (1 + np.sum(phi * mat_var.T, axis=1))

        # check output
        assert mean.shape == (x_q.shape[0],)
        assert var.shape == (x_q.shape[0],)
        return mean, var
