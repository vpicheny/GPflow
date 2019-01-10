# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..mean_functions import Zero

import numpy as np
import tensorflow as tf

from ..models.model import Model
from .. import kullback_leiblers, features
from .. import settings
from .. import transforms
from ..conditionals import conditional, Kuu
from ..decors import params_as_tensors
from ..models.model import GPModel
from ..params import DataHolder
from ..params import Minibatch
from ..params import Parameter, ParamList


class SVGPS(Model):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews,
                Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """

    def __init__(self, X, Y, kerns, likelihood, feats=None,
                 mean_functions=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 Zs=None,
                 num_data=None,
                 q_mus=None,
                 q_sqrts=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # sort out the X, Y into MiniBatch objects if required.
        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)
            
        self.kerns = kerns
        self.C = len(kerns)

        self.num_latent = num_latent or Y.shape[1]
        self.mean_functions = [ mean_functions[c] or Zero(output_dim=self.num_latent) for c in range(self.C)]
        self.likelihood = likelihood
        self.X, self.Y = X, Y

        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.features = [features.inducingpoint_wrapper(feat, Z) for feat,Z in zip(feats,  Zs)]

        # init variational parameters
        num_inducings = [len(feat) for feat in self.features]
        self._init_variational_parameters(num_inducings, q_mus, q_sqrts, q_diag)

    def _init_variational_parameters(self, num_inducings, q_mus, q_sqrts, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        C = len(num_inducings)
        q_mus = [np.zeros((num_inducings[c], self.num_latent)) if q_mus[c] is None else q_mus[c] for c in range(C)]
        self.q_mus = ParamList([ Parameter(q_mus[c], dtype=settings.float_type)  for c in range(C)]) # M x P

        if q_sqrts is None:
            if self.q_diag:
                self.q_sqrts = ParamList([ Parameter(np.ones((num_inducings[c], self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive) for c in range(C) ]) # M x P
            else:
                q_sqrts = [ np.array([np.eye(num_inducings[c], dtype=settings.float_type) for _ in range(self.num_latent)]) for c in range(C)]
                self.q_sqrts = ParamList([
                    Parameter(q_sqrts[c], transform=transforms.LowerTriangular(num_inducings[c], self.num_latent)) for c in range(C)
                ])
                # P x M x M
        else:
            if q_diag:
                for c in range(C):
                    assert q_sqrts[c].ndim == 2
                self.num_latent = q_sqrts[0].shape[1]
                self.q_sqrts = ParamList([
                    Parameter(q_sqrts[c], transform=transforms.positive) for c in range(C)
                ])# M x L/P
            else:
                for c in range(C):
                    assert q_sqrts[c].ndim == 3
                self.num_latent = q_sqrts[0].shape[0]
                num_inducings = [ q_sqrts[c].shape[1] for c in range(C) ]
                self.q_sqrts = ParamList([
                    Parameter(q_sqrts, transform=transforms.LowerTriangular(num_inducings[c], self.num_latent)) for c in range(C)
                ]) # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        KL = 0.
        for c in range(self.C):
            if self.whiten:
                Kc = None
            else:
                Kc = Kuu(self.features[c], self.kern[c], jitter=settings.numerics.jitter_level)  # (P x) x M x M

            KL += kullback_leiblers.gauss_kl(self.q_mus[c], self.q_sqrts[c], Kc)
        return KL

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(self.X)[0], settings.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):

        mus, vars = [], []
        for c in range(self.C):

            mu, var = conditional(Xnew, self.features[c], self.kerns[c], self.q_mus[c], q_sqrt=self.q_sqrts[c], full_cov=full_cov,
                                  white=self.whiten, full_output_cov=full_output_cov)
            mus.append(mu + self.mean_functions[c](Xnew))
            vars.append(var)

        return tf.stack(mus), tf.stack(vars)