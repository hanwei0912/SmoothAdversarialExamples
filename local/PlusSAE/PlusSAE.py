import warnings

import numpy as np
import tensorflow as tf
import pdb

from cleverhans.attacks import Attack
from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta
#from cleverhans.utils_tf import clip_eta, random_lp_vector
from cleverhans.compat import reduce_max, reduce_sum, softmax_cross_entropy_with_logits
from utils_SAE import CG

class SmoothPGDAttack(Attack):
    def __init__(self, model, sess=None, **kwargs):
        super(SmoothPGDAttack, self).__init__(model, sess=sess,
                                                    **kwargs)
        self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                                'clip_max')
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init', 'clip_grad',
                                  'sanity_checks']

    def optimize_linear(grad, eps, ord=np.inf):
        """
        Solves for the optimal input to a linear function under a norm constraint.
        Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)
        :param grad: tf tensor containing a batch of gradients
        :param eps: float scalar specifying size of constraint region
        :param ord: int specifying order of norm
        :returns:
          tf tensor containing optimal perturbation
        """

        # In Python 2, the `list` call in the following line is redundant / harmless.
        # In Python 3, the `list` call is needed to convert the iterator returned by `range` into a list.
        red_ind = list(range(1, len(grad.get_shape())))
        avoid_zero_div = 1e-12
        if ord == np.inf:
          # Take sign of gradient
          optimal_perturbation = tf.sign(grad)
          # The following line should not change the numerical results.
          # It applies only because `optimal_perturbation` is the output of
          # a `sign` op, which has zero derivative anyway.
          # It should not be applied for the other norms, where the
          # perturbation has a non-zero derivative.
          optimal_perturbation = tf.stop_gradient(optimal_perturbation)
        elif ord == 1:
          abs_grad = tf.abs(grad)
          sign = tf.sign(grad)
          max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
          tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
          num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
          optimal_perturbation = sign * tied_for_max / num_ties
        elif ord == 2:
          square = tf.maximum(avoid_zero_div,
                              reduce_sum(tf.square(grad),
                                         reduction_indices=red_ind,
                                         keepdims=True))
          optimal_perturbation = grad / tf.sqrt(square)
        else:
          raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                    "currently implemented.")

        # Scale perturbation to be the solution for the norm=eps rather than
        # norm=1 problem
        scaled_perturbation = utils_tf.mul(eps, optimal_perturbation)
        return scaled_perturbation


    def generate(self, x, A, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param kwargs: See `parse_params`
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        asserts = []

        # If a data range was specified, check that the input was in that range
        if self.clip_min is not None:
          asserts.append(utils_tf.assert_greater_equal(x,
                                                       tf.cast(self.clip_min,
                                                               x.dtype)))

        if self.clip_max is not None:
          asserts.append(utils_tf.assert_less_equal(x,
                                                    tf.cast(self.clip_max,
                                                            x.dtype)))

        # Initialize loop variables
        #if self.rand_init:
        #  eta = random_lp_vector(tf.shape(x), self.ord,
        #                         tf.cast(self.rand_init_eps, x.dtype),
        #                         dtype=x.dtype)
        #else:
        #  eta = tf.zeros(tf.shape(x))

        eta = tf.zeros(tf.shape(x))
        shape = x.shape
        batch_size = shape[0]
        #eta = tf.Variable(tf.zeros(tf.shape(x)))
        # Clip eta
        eta = clip_eta(eta, self.ord, self.eps)
        eta = CG(A, eta, shape)
        adv_x = x + eta
        logits = self.model.get_logits(adv_x)
        loss = softmax_cross_entropy_with_logits(labels=y,logits=logits)
        if targeted:
            loss = -loss
        if self.clip_min is not None or self.clip_max is not None:
          adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        if self.y_target is not None:
          y = self.y_target
          targeted = True
        elif self.y is not None:
          y = self.y
          targeted = False
        else:
          model_preds = self.model.get_probs(x)
          preds_max = tf.reduce_max(model_preds, 1)
          y = tf.to_float(tf.equal(model_preds, preds_max))
          y = tf.stop_gradient(y)
          targeted = False
          del model_preds

        y_kwarg = 'y_target' if targeted else 'y'

        def cond(i, _):
          """Iterate until requested number of iterations is completed"""
          return tf.less(i, self.nb_iter)

        def body(i, eta):
          """Do a projected gradient step"""
          grad, = tf.gradients(loss, eta)
          grad = CG(A, grad, shape)
          grad = clip_eta(grad, self.ord, self.eps)
          #if clip_grad:
          #    grad = utils_tf.zero_out_clipped_grads(grad, x, clip_min, clip_max)
          #optimal_perturbation = self.optimize_linear(grad, self.eps, ord=self.ord)
          adv_x = x + eta +grad
          if (self.clip_min is not None) or (self.clip_max is not None):
              # We don't currently support one-sided clipping
              assert self.clip_min is not None and self.clip_max is not None
              adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
          # Clipping perturbation eta to self.ord norm ball
          eta = adv_x - x
          return i + 1, eta

        _, eta = tf.while_loop(cond, body, (tf.zeros([]), eta), back_prop=True)

        # Asserts run only on CPU.
        # When multi-GPU eval code tries to force all PGD ops onto GPU, this
        # can cause an error.
        common_dtype = tf.float32
        asserts.append(utils_tf.assert_less_equal(tf.cast(self.eps_iter,
                                                          dtype=common_dtype),
                                                  tf.cast(self.eps, dtype=common_dtype)))
        if self.ord == np.inf and self.clip_min is not None:
          # The 1e-6 is needed to compensate for numerical error.
          # Without the 1e-6 this fails when e.g. eps=.2, clip_min=.5,
          # clip_max=.7
          asserts.append(utils_tf.assert_less_equal(tf.cast(self.eps, x.dtype),
                                                    1e-6 + tf.cast(self.clip_max,
                                                                   x.dtype)
                                                    - tf.cast(self.clip_min,
                                                              x.dtype)))

        if self.sanity_checks:
          with tf.control_dependencies(asserts):
            adv_x = tf.identity(adv_x)

        return adv_x

    def parse_params(self,
                     eps=0.3,
                     eps_iter=0.05,
                     nb_iter=10,
                     y=None,
                     ord=np.inf,
                     clip_min=None,
                     clip_max=None,
                     y_target=None,
                     rand_init=None,
                     rand_init_eps=None,
                     clip_grad=False,
                     sanity_checks=True,
                     **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (optional float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (optional float) step size for each attack iteration
        :param nb_iter: (optional int) Number of attack iterations.
        :param y: (optional) A tensor with the true labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional) Start the gradient descent from a point chosen
                          uniformly at random in the norm ball of radius
                          rand_init_eps
        :param rand_init_eps: (optional float) size of the norm ball from which
                              the initial starting point is chosen. Defaults to eps
        :param clip_grad: (optional bool) Ignore gradient components at positions
                          where the input is already at the boundary of the domain,
                          and the update step will get clipped out.
        :param sanity_checks: bool Insert tf asserts checking values
            (Some tests need to run with no sanity checks because the
             tests intentionally configure the attack strangely)
        """

        # Save attack-specific parameters
        self.eps = eps
        if rand_init is None:
          rand_init = True
        self.rand_init = rand_init
        if rand_init_eps is None:
          rand_init_eps = self.eps
        self.rand_init_eps = rand_init_eps

        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_grad = clip_grad

        if isinstance(eps, float) and isinstance(eps_iter, float):
          # If these are both known at compile time, we can check before anything
          # is run. If they are tf, we can't check them yet.
          assert eps_iter <= eps, (eps_iter, eps)

        if self.y is not None and self.y_target is not None:
          raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
          raise ValueError("Norm order must be either np.inf, 1, or 2.")

        if self.clip_grad and (self.clip_min is None or self.clip_max is None):
          raise ValueError("Must set clip_min and clip_max if clip_grad is set")

        self.sanity_checks = sanity_checks

        if len(kwargs.keys()) > 0:
          warnings.warn("kwargs is unused and will be removed on or after "
                        "2019-04-26.")

        return True

