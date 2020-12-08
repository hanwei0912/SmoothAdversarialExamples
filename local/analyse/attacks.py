from abc import ABCMeta
import numpy as np
from six.moves import xrange
import warnings
import collections
import pdb

import cleverhans.utils as utils
from cleverhans.model import Model, CallableModelWrapper

_logger = utils.create_logger("cleverhans.attacks.hw")


class Attack(object):

    """
    Abstract base class for all attack classes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, back='tf', sess=None):
        """
        :param model: An instance of the cleverhans.model.Model class.
        :param back: The backend to use. Currently 'tf' is the only option.
        :param sess: The tf session to run graphs in
        """
        if not(back == 'tf'):
            raise ValueError("Backend argument must either be 'tf'.")

        if back == 'tf' and sess is None:
            import tensorflow as tf
            sess = tf.get_default_session()

        if not isinstance(model, Model):
            raise ValueError("The model argument should be an instance of"
                             " the cleverhans.model.Model class.")

        # Prepare attributes
        self.model = model
        self.back = back
        self.sess = sess


        # We are going to keep track of old graphs and cache them.
        self.graphs = {}

        # When calling generate_np, arguments in the following set should be
        # fed into the graph, as they are not structural items that require
        # generating a new graph.
        # This dict should map names of arguments to the types they should
        # have.
        # (Usually, the target class will be a feedable keyword argument.)
        self.feedable_kwargs = {}

        # When calling generate_np, arguments in the following set should NOT
        # be fed into the graph, as they ARE structural items that require
        # generating a new graph.
        # This list should contain the names of the structural arguments.
        self.structural_kwargs = []

    def generate(self, x, **kwargs):
        """
        Generate the attack's symbolic graph for adversarial examples. This
        method should be overriden in any child class that implements an
        attack that is expressable symbolically. Otherwise, it will wrap the
        numerical implementation as a symbolic operator.
        :param x: The model's symbolic inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A symbolic representation of the adversarial examples.
        """

        error = "Sub-classes must implement generate."
        raise NotImplementedError(error)

    def construct_graph(self, fixed, feedable, x_val, hash_key):
        """
        Construct the graph required to run the attack through generate_np.
        :param fixed: Structural elements that require defining a new graph.
        :param feedable: Arguments that can be fed to the same graph when
                         they take different values.
        :param x_val: symbolic adversarial example
        :param hash_key: the key used to store this graph in our cache
        """
        # try our very best to create a TF placeholder for each of the
        # feedable keyword arguments, and check the types are one of
        # the allowed types
        import tensorflow as tf

        class_name = str(self.__class__).split(".")[-1][:-2]
        _logger.info("Constructing new graph for attack " + class_name)

        # remove the None arguments, they are just left blank
        for k in list(feedable.keys()):
            if feedable[k] is None:
                del feedable[k]

        # process all of the rest and create placeholders for them
        new_kwargs = dict(x for x in fixed.items())
        for name, value in feedable.items():
            given_type = self.feedable_kwargs[name]
            if isinstance(value, np.ndarray):
                new_shape = [None] + list(value.shape[1:])
                new_kwargs[name] = tf.placeholder(given_type, new_shape)
            elif isinstance(value, utils.known_number_types):
                new_kwargs[name] = tf.placeholder(given_type, shape=[])
            else:
                raise ValueError("Could not identify type of argument " +
                                 name + ": " + str(value))

        # x is a special placeholder we always want to have
        x_shape = [None] + list(x_val.shape)[1:]
        x = tf.placeholder(tf.float32, shape=x_shape)

        # now we generate the graph that we want
        x_adv = self.generate(x, **new_kwargs)

        self.graphs[hash_key] = (x, new_kwargs, x_adv)

        if len(self.graphs) >= 10:
            warnings.warn("Calling generate_np() with multiple different "
                          "structural paramaters is inefficient and should"
                          " be avoided. Calling generate() is preferred.")

    def generate_np(self, x_val, **kwargs):
        """
        Generate adversarial examples and return them as a NumPy array.
        Sub-classes *should not* implement this method unless they must
        perform special handling of arguments.
        :param x_val: A NumPy array with the original inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A NumPy array holding the adversarial examples.
        """
        if self.sess is None:
            raise ValueError("Cannot use `generate_np` when no `sess` was"
                             " provided")

        fixed, feedable, hash_key = self.construct_variables(kwargs)

        # for loop, del the none feedable variable
        for k in list(feedable.keys()):
            if feedable[k] is None:
                del feedable[k]

        if hash_key not in self.graphs:
            self.construct_graph(fixed, feedable, x_val, hash_key)

        x, new_kwargs, x_adv = self.graphs[hash_key]

        feed_dict = {x: x_val}

        for name in feedable:
            feed_dict[new_kwargs[name]] = feedable[name]

        return self.sess.run(x_adv, feed_dict)


    def construct_variables(self, kwargs):
        """
        Construct the inputs to the attack graph to be used by generate_np.
        :param kwargs: Keyword arguments to generate_np.
        :return: Structural and feedable arguments as well as a unique key
                 for the graph given these inputs.
        """
        # the set of arguments that are structural properties of the attack
        # if these arguments are different, we must construct a new graph
        fixed = dict((k, v) for k, v in kwargs.items()
                     if k in self.structural_kwargs)

        # the set of arguments that are passed as placeholders to the graph
        # on each call, and can change without constructing a new graph
        feedable = dict((k, v) for k, v in kwargs.items()
                        if k in self.feedable_kwargs)

        if len(fixed) + len(feedable) < len(kwargs):
            warnings.warn("Supplied extra keyword arguments that are not "
                          "used in the graph computation. They have been "
                          "ignored.")

        if not all(isinstance(value, collections.Hashable)
                   for value in fixed.values()):
            # we have received a fixed value that isn't hashable
            # this means we can't cache this graph for later use,
            # and it will have to be discarded later
            hash_key = None
        else:
            # create a unique key for this set of fixed paramaters
            hash_key = tuple(sorted(fixed.items()))

        return fixed, feedable, hash_key

    def get_or_guess_labels(self, x, kwargs):
        """
        Get the label to use in generating an adversarial example for x.
        The kwargs are fed directly from the kwargs of the attack.
        If 'y' is in kwargs, then assume it's an untargeted attack and
        use that as the label.
        If 'y_target' is in kwargs, then assume it's a targeted attack and
        use that as the label.
        Otherwise, use the model's prediction as the label and perform an
        untargeted attack.
        """
        import tensorflow as tf

        if 'y' in kwargs and 'y_target' in kwargs:
            raise ValueError("Can not set both 'y' and 'y_target'.")
        elif 'y' in kwargs:
            labels = kwargs['y']
        elif 'y_target' in kwargs:
            labels = kwargs['y_target']
        else:
            preds = self.model.get_probs(x)
            preds_max = tf.reduce_max(preds, 1, keep_dims=True)
            original_predictions = tf.to_float(tf.equal(preds,
                                                        preds_max))
            labels = tf.stop_gradient(original_predictions)
        if isinstance(labels, np.ndarray):
            nb_classes = labels.shape[1]
        else:
            nb_classes = labels.get_shape().as_list()[1]
        return labels, nb_classes

    def parse_params(self, params=None):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        :param params: a dictionary of attack-specific parameters
        :return: True when parsing was successful
        """
        return True

class FastGradientMethod(Attack):

    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This
    implementation extends the attack to other norms, and is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a FastGradientMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(FastGradientMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        from attacks_tf import fgm

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        return fgm(x, self.model.get_probs(x), y=labels, eps=self.eps,
                   ord=self.ord, clip_min=self.clip_min,
                   clip_max=self.clip_max,
                   targeted=(self.y_target is not None))

    def parse_params(self, eps=0.3, ord=np.inf, y=None, y_target=None,
                     clip_min=None, clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters

        self.eps = eps
        self.ord = ord
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2),int(0)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        return True


class BasicIterativeMethod(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(BasicIterativeMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']
        self.norm=0

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate(x + eta, **fgm_params) - x

            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm
                self.norm =eta

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

class TeddyIterativeMethod(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(TeddyIterativeMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = tf.zeros_like(x)
        adv_x = x

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        def while_condition_stop(eta,adv_x):
            delta = adv_x - x
            reduc_ind = list(xrange(1, len(delta.get_shape())))
            norm = tf.sqrt(tf.reduce_sum(tf.square(delta),
                        reduction_indices=reduc_ind,
                        keep_dims=True))
            cond2_sum=tf.less(norm,tf.ones_like(norm)*self.eps)
            cond2 = tf.reduce_all(cond2_sum)
            return cond2


        def big_body(eta,adv_x):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            adv_x = adv_x + eta
            eta = FGM.generate( adv_x , **fgm_params) - adv_x

            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm
            return eta, adv_x

        eta, adv_x = tf.while_loop(while_condition_stop,big_body,[eta,adv_x])
        # Define adversarial example (and clip if necessary)
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

class SmoothTeddyIterativeMethod(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(SmoothTeddyIterativeMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, A, At, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = tf.zeros_like(x)
        adv_x = x

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        def while_condition_stop(eta,adv_x):
            delta = adv_x - x
            reduc_ind = list(xrange(1, len(delta.get_shape())))
            norm = tf.sqrt(tf.reduce_sum(tf.square(delta),
                        reduction_indices=reduc_ind,
                        keep_dims=True))
            cond2_sum=tf.less(norm,tf.ones_like(norm)*self.eps)
            cond2 = tf.reduce_all(cond2_sum)
            return cond2


        def big_body(eta,adv_x):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            adv_x = adv_x + eta
            eta = FGM.generate( adv_x , **fgm_params) - adv_x
            shape = eta.shape
            eta_r = tf.reshape(eta,[-1,1,784])
            eta_r = tf.matmul(eta_r,A)
            eta_r = tf.matmul(eta_r,At)
            eta = tf.reshape(eta_r,[-1,28,28,1])

            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm
            return eta, adv_x

        eta, adv_x = tf.while_loop(while_condition_stop,big_body,[eta,adv_x])
        # Define adversarial example (and clip if necessary)
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True


class SmoothTeddyIterativeMethodCG(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(SmoothTeddyIterativeMethodCG, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, Aa, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = tf.zeros_like(x)
        adv_x = x
        A = Aa
        self.A =A
        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        def while_condition_stop(eta,adv_x):
            delta = adv_x - x
            reduc_ind = list(xrange(1, len(delta.get_shape())))
            norm = tf.sqrt(tf.reduce_sum(tf.square(delta),
                        reduction_indices=reduc_ind,
                        keep_dims=True))
            cond2_sum=tf.less(norm,tf.ones_like(norm)*self.eps)
            cond2 = tf.reduce_any(cond2_sum)
            return cond2


        def big_body(eta,adv_x):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            adv_x = adv_x + eta
            eta = FGM.generate( adv_x , **fgm_params) - adv_x

            shape = eta.shape
            modifier = tf.reshape(eta,(shape[0],shape[1]*shape[2],shape[3]))
            def multi_sparse(mod, A, batch_size, shape):
                im_mod = tf.reshape(
                    mod, (batch_size, shape[1], shape[2],shape[3]))
                im_mod = tf.pad(im_mod, paddings=[[0, 0], [1, 1], [
                                1, 1], [0, 0]], mode="CONSTANT")
                im_r = tf.slice(im_mod, [0, 0, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_l = tf.slice(im_mod, [0, 2, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_u = tf.slice(im_mod, [0, 1, 0, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_d = tf.slice(im_mod, [0, 1, 2, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_r = tf.reshape(im_r, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_l = tf.reshape(im_l, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_u = tf.reshape(im_u, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_d = tf.reshape(im_d, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_a = tf.concat([im_r, im_l, im_u, im_d], 1)
                A = tf.reshape(A, (batch_size, 4, shape[1]*shape[2], shape[3]))
                smo = tf.multiply(A, im_a)
                smo = tf.reduce_sum(smo, axis=1)
                smo = tf.reshape(smo, (batch_size, shape[1]*shape[2], shape[3]))
                smo_t = tf.add(smo, mod)
                return smo_t

            def while_condition(i, r, p, s, x): return tf.less(i, 50)

            def body(i, r, p, s, x):
                q = multi_sparse(p, self.A, shape[0], shape)
                alpha_t = tf.div(s, tf.matmul(tf.transpose(p, [0, 2, 1]),q))
                alpha_t = tf.map_fn(lambda x: tf.diag_part(x),alpha_t,dtype=tf.float32)
                alpha =tf.tile(tf.reshape(alpha_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                x = tf.add(x, tf.multiply(alpha, p))
                r = tf.subtract(r, tf.multiply(alpha, q))
                t = tf.matmul(tf.transpose(r, [0, 2, 1]),r)
                beta_t = tf.div(t, s)
                beta_t = tf.map_fn(lambda x: tf.diag_part(x),beta_t,dtype=tf.float32)
                beta = tf.tile(tf.reshape(beta_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                p = tf.add(r, tf.multiply(beta, p))
                s = t
                i = tf.add(i, 1)
                return i, r, p, s, x

            # zeros situation
            nn = tf.matmul(tf.transpose(modifier, [0, 2, 1]), modifier)
            oo = tf.zeros_like(nn)
            noeq = tf.equal(nn, oo)
            noeq_int = tf.to_int32(noeq)
            noeq_res = tf.equal(tf.reduce_sum(noeq_int), tf.reduce_sum(tf.ones_like(noeq_int)))

            def f_false(modifier):
                i = tf.constant(0)
                r = modifier - multi_sparse(modifier, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = modifier
                i, r, p, s, smo_mod = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])
                ## norm
                z_d = tf.ones_like(modifier)
                i = tf.constant(0)
                r = z_d - multi_sparse(z_d, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = z_d
                i, r, p, s, div_z = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])

                smo_mod = tf.div(smo_mod, div_z)
                smo_mod = tf.reshape(smo_mod, shape)
                return smo_mod

            def f_true(mm):
                smo_mod = tf.reshape(mm,shape)
                return smo_mod

            smo_mod = tf.cond(noeq_res, lambda: f_true(modifier),lambda: f_false(modifier))

            self.modifier =tf.reshape(modifier,shape)
            eta = smo_mod

            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm
            return eta, adv_x

        eta, adv_x = tf.while_loop(while_condition_stop,big_body,[eta,adv_x])
        # Define adversarial example (and clip if necessary)
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True



class MomentumIterativeMethod(Attack):

    """
    The Momentum Iterative Method (Dong et al. 2017). This method won
    the first places in NIPS 2017 Non-targeted Adversarial Attacks and
    Targeted Adversarial Attacks. The original paper used hard labels
    for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a MomentumIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(MomentumIterativeMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter', 'decay_factor']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param decay_factor: (optional) Decay factor for the momentum term.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        momentum = 0
        adv_x = x

        # Fix labels to the first model predictions for loss computation
        y, nb_classes = self.get_or_guess_labels(x, kwargs)
        y = y / tf.reduce_sum(y, 1, keep_dims=True)
        targeted = (self.y_target is not None)

        from . import utils_tf
        for i in range(self.nb_iter):
            # Compute loss
            preds = self.model.get_probs(adv_x)
            loss = utils_tf.model_loss(y, preds, mean=False)
            if targeted:
                loss = -loss

            # Define gradient of loss wrt input
            grad, = tf.gradients(loss, adv_x)

            # Normalize current gradient and add it to the accumulated gradient
            red_ind = list(xrange(1, len(grad.get_shape())))
            avoid_zero_div = 1e-12
            grad = grad / tf.maximum(avoid_zero_div,
                                     tf.reduce_mean(tf.abs(grad),
                                                    red_ind,
                                                    keep_dims=True))
            momentum = self.decay_factor * momentum + grad

            if self.ord == np.inf:
                normalized_grad = tf.sign(momentum)
            elif self.ord == 1:
                norm = tf.maximum(avoid_zero_div,
                                  tf.reduce_sum(tf.abs(momentum),
                                                red_ind,
                                                keep_dims=True))
                normalized_grad = momentum / norm
            elif self.ord == 2:
                square = tf.reduce_sum(tf.square(momentum),
                                       red_ind,
                                       keep_dims=True)
                norm = tf.sqrt(tf.maximum(avoid_zero_div, square))
                normalized_grad = momentum / norm
            else:
                raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                          "currently implemented.")

            # Update and clip adversarial example in current iteration
            scaled_grad = self.eps_iter * normalized_grad
            adv_x = adv_x + scaled_grad
            adv_x = x + utils_tf.clip_eta(adv_x - x, self.ord, self.eps)

            if self.clip_min is not None and self.clip_max is not None:
                adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

            adv_x = tf.stop_gradient(adv_x)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.06, nb_iter=10, y=None,
                     ord=np.inf, decay_factor=1.0,
                     clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param decay_factor: (optional) Decay factor for the momentum term.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.decay_factor = decay_factor
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True


class SaliencyMapMethod(Attack):

    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a SaliencyMapMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(SaliencyMapMethod, self).__init__(model, back, sess)

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

        import tensorflow as tf
        self.feedable_kwargs = {'y_target': tf.float32}
        self.structural_kwargs = ['theta', 'gamma',
                                  'clip_max', 'clip_min', 'symbolic_impl']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param theta: (optional float) Perturbation introduced to modified
                      components (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features
        :param clip_min: (optional float) Minimum component value for clipping
        :param clip_max: (optional float) Maximum component value for clipping
        :param y_target: (optional) Target tensor if the attack is targeted
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        if self.symbolic_impl:
            from .attacks_tf import jsma_symbolic

            # Create random targets if y_target not provided
            if self.y_target is None:
                from random import randint

                def random_targets(gt):
                    result = gt.copy()
                    nb_s = gt.shape[0]
                    nb_classes = gt.shape[1]

                    for i in xrange(nb_s):
                        result[i, :] = np.roll(result[i, :],
                                               randint(1, nb_classes-1))

                    return result

                labels, nb_classes = self.get_or_guess_labels(x, kwargs)
                self.y_target = tf.py_func(random_targets, [labels],
                                           tf.float32)
                self.y_target.set_shape([None, nb_classes])

            x_adv = jsma_symbolic(x, model=self.model, y_target=self.y_target,
                                  theta=self.theta, gamma=self.gamma,
                                  clip_min=self.clip_min,
                                  clip_max=self.clip_max)
        else:
            from .attacks_tf import jacobian_graph, jsma_batch

            # Define Jacobian graph wrt to this input placeholder
            preds = self.model.get_probs(x)
            nb_classes = preds.get_shape().as_list()[-1]
            grads = jacobian_graph(preds, x, nb_classes)

            # Define appropriate graph (targeted / random target labels)
            if self.y_target is not None:
                def jsma_wrap(x_val, y_target):
                    return jsma_batch(self.sess, x, preds, grads, x_val,
                                      self.theta, self.gamma, self.clip_min,
                                      self.clip_max, nb_classes,
                                      y_target=y_target)

                # Attack is targeted, target placeholder will need to be fed
                x_adv = tf.py_func(jsma_wrap, [x, self.y_target], tf.float32)
            else:
                def jsma_wrap(x_val):
                    return jsma_batch(self.sess, x, preds, grads, x_val,
                                      self.theta, self.gamma, self.clip_min,
                                      self.clip_max, nb_classes,
                                      y_target=None)

                # Attack is untargeted, target values will be chosen at random
                x_adv = tf.py_func(jsma_wrap, [x], tf.float32)

        return x_adv

    def parse_params(self, theta=1., gamma=1., nb_classes=None,
                     clip_min=0., clip_max=1., y_target=None,
                     symbolic_impl=True, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param theta: (optional float) Perturbation introduced to modified
                      components (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features
        :param nb_classes: (optional int) Number of model output classes
        :param clip_min: (optional float) Minimum component value for clipping
        :param clip_max: (optional float) Maximum component value for clipping
        :param y_target: (optional) Target tensor if the attack is targeted
        """

        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y_target = y_target
        self.symbolic_impl = symbolic_impl

        return True


class VirtualAdversarialMethod(Attack):

    """
    This attack was originally proposed by Miyato et al. (2016) and was used
    for virtual adversarial training.
    Paper link: https://arxiv.org/abs/1507.00677
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(VirtualAdversarialMethod, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'eps': tf.float32, 'xi': tf.float32,
                                'clip_min': tf.float32,
                                'clip_max': tf.float32}
        self.structural_kwargs = ['num_iterations']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float ) the epsilon (input variation parameter)
        :param num_iterations: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        return vatm(self.model, x, self.model.get_logits(x), eps=self.eps,
                    num_iterations=self.num_iterations, xi=self.xi,
                    clip_min=self.clip_min, clip_max=self.clip_max)

    def parse_params(self, eps=2.0, num_iterations=1, xi=1e-6, clip_min=None,
                     clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (optional float )the epsilon (input variation parameter)
        :param num_iterations: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters
        self.eps = eps
        self.num_iterations = num_iterations
        self.xi = xi
        self.clip_min = clip_min
        self.clip_max = clip_max
        return True

class CarliniWagnerL2(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(CarliniWagnerL2, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from attacks_tf import CarliniWagnerL2 as CWL2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CWL2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val):
            a,b = attack.attack(x_val,y_val)
            return np.array(a, dtype=np.float32), np.array(b,dtype=np.float32)
            #return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap,b = tf.py_func(cw_wrap, [x, labels], (tf.float32,tf.float32))

        return wrap,b

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

class CarliniWagnerLi(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        super(CarliniWagnerLi, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['targeted', 'learning_rate',
                                  'max_iterations', 'abort_early',
                                  'initial_const', 'largest_const',
                                  'reduce_const', 'decrease_factor',
                                  'const_factor']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import CarliniWagnerLi as CWLi
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        #pdb.set_trace()
        attack = CWLi(self.sess, self.model,
                     'y_target' in kwargs, self.learning_rate, 
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.largest_const,
                      self.reduce_const, self.decrease_factor,
                      self.const_factor,
                      nb_classes, x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap = tf.py_func(cw_wrap, [x, labels], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     learning_rate=5e-3,max_iterations=1000,
                     abort_early=True, initial_const=1e-5,
                     largest_const = 2e+1, reduce_const = False,
                     decrease_factor = 0.9, const_factor = 2.0):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.reduce_const = reduce_const
        self.decrease_factor = decrease_factor
        self.const_factor = const_factor


class ElasticNetMethod(Attack):
    """
    This attack features L1-oriented adversarial examples and includes
    the C&W L2 attack as a special case (when beta is set to 0).
    Adversarial examples attain similar performance to those
    generated by the C&W L2 attack, and more importantly,
    have improved transferability properties and
    complement adversarial training.
    Paper link: https://arxiv.org/abs/1709.04114
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(ElasticNetMethod, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['beta', 'batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param beta: Trades off L2 distortion with L1 distortion: higher
                     produces examples with lower L1 distortion, at the
                     cost of higher L2 (and typically Linf) distortion
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the perturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early abort when the total
                            loss starts to increase (greatly speeds up attack,
                            but hurts performance, particularly on ImageNet)
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the perturbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        self.parse_params(**kwargs)

        from .attacks_tf import ElasticNetMethod as EAD
        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = EAD(self.sess, self.model, self.beta,
                     self.batch_size, self.confidence,
                     'y_target' in kwargs, self.learning_rate,
                     self.binary_search_steps, self.max_iterations,
                     self.abort_early, self.initial_const,
                     self.clip_min, self.clip_max,
                     nb_classes, x.get_shape().as_list()[1:])

        def ead_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap = tf.py_func(ead_wrap, [x, labels], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None,
                     nb_classes=None, beta=1e-3,
                     batch_size=1, confidence=0,
                     learning_rate=1e-2,
                     binary_search_steps=9, max_iterations=1000,
                     abort_early=False, initial_const=1e-3,
                     clip_min=0, clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.beta = beta
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max


class DeepFool(Attack):

    """
    DeepFool is an untargeted & iterative attack which is based on an
    iterative linearization of the classifier. The implementation here
    is w.r.t. the L2 norm.
    Paper link: "https://arxiv.org/pdf/1511.04599.pdf"
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a DeepFool instance.
        """
        super(DeepFool, self).__init__(model, back, sess)

        self.structural_kwargs = ['over_shoot', 'max_iter', 'clip_max',
                                  'clip_min', 'nb_candidate']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param nb_candidate: The number of classes to test against, i.e.,
                             deepfool only consider nb_candidate classes when
                             attacking(thus accelerate speed). The nb_candidate
                             classes are chosen according to the prediction
                             confidence during implementation.
        :param overshoot: A termination criterion to prevent vanishing updates
        :param max_iter: Maximum number of iteration for deepfool
        :param nb_classes: The number of model output classes
        :param clip_min: Minimum component value for clipping
        :param clip_max: Maximum component value for clipping
        """

        import tensorflow as tf
        from .attacks_tf import jacobian_graph, deepfool_batch

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Define graph wrt to this input placeholder
        logits = self.model.get_logits(x)
        self.nb_classes = logits.get_shape().as_list()[-1]
        assert self.nb_candidate <= self.nb_classes,\
            'nb_candidate should not be greater than nb_classes'
        preds = tf.reshape(tf.nn.top_k(logits, k=self.nb_candidate)[0],
                           [-1, self.nb_candidate])
        # grads will be the shape [batch_size, nb_candidate, image_size]
        grads = tf.stack(jacobian_graph(preds, x, self.nb_candidate), axis=1)

        # Define graph
        def deepfool_wrap(x_val):
            return deepfool_batch(self.sess, x, preds, logits, grads, x_val,
                                  self.nb_candidate, self.overshoot,
                                  self.max_iter, self.clip_min, self.clip_max,
                                  self.nb_classes)
        return tf.py_func(deepfool_wrap, [x], tf.float32)

    def parse_params(self, nb_candidate=10, overshoot=0.02, max_iter=50,
                     nb_classes=None, clip_min=0., clip_max=1., **kwargs):
        """
        :param nb_candidate: The number of classes to test against, i.e.,
                             deepfool only consider nb_candidate classes when
                             attacking(thus accelerate speed). The nb_candidate
                             classes are chosen according to the prediction
                             confidence during implementation.
        :param overshoot: A termination criterion to prevent vanishing updates
        :param max_iter: Maximum number of iteration for deepfool
        :param nb_classes: The number of model output classes
        :param clip_min: Minimum component value for clipping
        :param clip_max: Maximum component value for clipping
        """
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.nb_candidate = nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max

        return True


def vatm(model, x, logits, eps, back='tf', num_iterations=1, xi=1e-6,
         clip_min=None, clip_max=None):
    """
    A wrapper for the perturbation methods used for virtual adversarial
    training : https://arxiv.org/abs/1507.00677
    It calls the right function, depending on the
    user's backend.
    :param model: the model which returns the network unnormalized logits
    :param x: the input placeholder
    :param logits: the model's unnormalized output tensor
    :param eps: the epsilon (input variation parameter)
    :param num_iterations: the number of iterations
    :param xi: the finite difference parameter
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """
    assert back == 'tf'
    # Compute VATM using TensorFlow
    from .attacks_tf import vatm as vatm_tf
    return vatm_tf(model, x, logits, eps, num_iterations=num_iterations,
                   xi=xi, clip_min=clip_min, clip_max=clip_max)


class MadryEtAl(Attack):

    """
    The Projected Gradient Descent Attack (Madry et al. 2017).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a MadryEtAl instance.
        """
        super(MadryEtAl, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        # Initialize loop variables
        adv_x = self.attack(x, labels)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.01, nb_iter=40, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, rand_init=True, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rand_init = rand_init

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import model_loss, clip_eta

        adv_x = x + eta
        preds = self.model.get_probs(adv_x)
        loss = model_loss(y, preds)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return eta

    def attack(self, x, y):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.
        :param x: A tensor with the input image.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        if self.rand_init:
            eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
            eta = clip_eta(eta, self.ord, self.eps)
        else:
            eta = tf.zeros_like(x)

        for i in range(self.nb_iter):
            eta = self.attack_single_step(x, eta, y)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x


class FastFeatureAdversaries(Attack):
    """
    This is a fast implementation of "Feature Adversaries", an attack
    against a target internal representation of a model.
    "Feature adversaries" were originally introduced in (Sabour et al. 2016),
    where the optimization was done using LBFGS.
    Paper link: https://arxiv.org/abs/1511.05122
    This implementation is similar to "Basic Iterative Method"
    (Kurakin et al. 2016) but applied to the internal representations.
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a FastFeatureAdversaries instance.
        """
        super(FastFeatureAdversaries, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32,
                                'layer': str}
        self.structural_kwargs = ['ord', 'nb_iter']

        assert isinstance(self.model, Model)

    def parse_params(self, layer=None, eps=0.3, eps_iter=0.05, nb_iter=10,
                     ord=np.inf, clip_min=None, clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param layer: (required str) name of the layer to target.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.layer = layer
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

    def attack_single_step(self, x, eta, g_feat):
        """
        TensorFlow implementation of the Fast Feature Gradient. This is a
        single step attack similar to Fast Gradient Method that attacks an
        internal representation.
        :param x: the input placeholder
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param g_feat: model's internal tensor for guide
        :return: a tensor for the adversarial example
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        adv_x = x + eta
        a_feat = self.model.get_layer(adv_x, self.layer)

        # feat.shape = (batch, c) or (batch, w, h, c)
        axis = list(range(1, len(a_feat.shape)))

        # Compute loss
        # This is a targeted attack, hence the negative sign
        loss = -tf.reduce_sum(tf.square(a_feat - g_feat), axis)

        # Define gradient of loss wrt input
        grad, = tf.gradients(loss, adv_x)

        # Multiply by constant epsilon
        scaled_signed_grad = self.eps_iter * tf.sign(grad)

        # Add perturbation to original example to obtain adversarial example
        adv_x = adv_x + scaled_signed_grad

        # If clipping is needed,
        # reset all values outside of [clip_min, clip_max]
        if (self.clip_min is not None) and (self.clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        adv_x = tf.stop_gradient(adv_x)

        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)

        return eta

    def generate(self, x, g, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param g: The target's symbolic representation.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        g_feat = self.model.get_layer(g, self.layer)

        # Initialize loop variables
        eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
        eta = clip_eta(eta, self.ord, self.eps)

        for i in range(self.nb_iter):
            eta = self.attack_single_step(x, eta, g_feat)

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

class LBFGS(Attack):
    """
    LBFGS is the first adversarial attack for convolutional neural networks,
    and is a target & iterative attack.
    Paper link: "https://arxiv.org/pdf/1312.6199.pdf"
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(LBFGS, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y_target': tf.float32}
        self.structural_kwargs = ['batch_size', 'binary_search_steps',
                                  'max_iterations', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y_target: (required) A tensor with the one-hot target labels.
        :param batch_size: The number of inputs to include in a batch and
                           process simultaneously.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and cross-entropy loss of classification.
        :param max_iterations: The maximum number of iterations.
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the perturbation
                              and cross-entropy loss of the classification.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import LBFGS_attack
        self.parse_params(**kwargs)

        _, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = LBFGS_attack(self.sess, x, self.model.get_probs(x),
                              self.y_target, self.binary_search_steps,
                              self.max_iterations, self.initial_const,
                              self.clip_min, self.clip_max, nb_classes,
                              self.batch_size)

        def lbfgs_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap = tf.py_func(lbfgs_wrap, [x, self.y_target], tf.float32)

        return wrap

    def parse_params(self, y_target=None, batch_size=1,
                     binary_search_steps=5, max_iterations=1000,
                     initial_const=1e-2, clip_min=0, clip_max=1):

        self.y_target = y_target
        self.batch_size = batch_size
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

class Clip_version(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_version, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max','A']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Clip_version as CLV2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CLV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:],self.A)

        def cv_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap = tf.py_func(cv_wrap, [x, labels], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1,A=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.A = A

class Sigmoid_version(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Sigmoid_version, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max','A']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Sigmoid_version as SV2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        
        attack = SV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:],self.A)

        def cv_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap = tf.py_func(cv_wrap, [x, labels], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1,A=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.A = A

class SmoothBasicIterativeMethod(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(SmoothBasicIterativeMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, A, At, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        #A = tf.placeholder(tf.float32, shape=(None,784,784))
        #At = tf.placeholder(tf.float32,shape=(None,784,784))
        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate(x + eta, **fgm_params) - x

            shape = eta.shape
            eta_r = tf.reshape(eta,[-1,1,784])
            eta_r = tf.matmul(eta_r,A)
            eta_r = tf.matmul(eta_r,At)
            eta = tf.reshape(eta_r,[-1,28,28,1])

            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True


class SmoothBasicIterativeMethodCG_2(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(SmoothBasicIterativeMethodCG_2, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, Aa, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        A = Aa
        self.A=A
        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate(x + eta, **fgm_params) - x
            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm

            shape = eta.shape
            modifier = tf.reshape(eta,(shape[0],shape[1]*shape[2],shape[3]))
            def multi_sparse(mod, A, batch_size, shape):
                im_mod = tf.reshape(
                    mod, (batch_size, shape[1], shape[2],shape[3]))
                im_mod = tf.pad(im_mod, paddings=[[0, 0], [1, 1], [
                                1, 1], [0, 0]], mode="CONSTANT")
                im_r = tf.slice(im_mod, [0, 0, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_l = tf.slice(im_mod, [0, 2, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_u = tf.slice(im_mod, [0, 1, 0, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_d = tf.slice(im_mod, [0, 1, 2, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_r = tf.reshape(im_r, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_l = tf.reshape(im_l, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_u = tf.reshape(im_u, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_d = tf.reshape(im_d, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_a = tf.concat([im_r, im_l, im_u, im_d], 1)
                A = tf.reshape(A, (batch_size, 4, shape[1]*shape[2], shape[3]))
                smo = tf.multiply(A, im_a)
                smo = tf.reduce_sum(smo, axis=1)
                smo = tf.reshape(smo, (batch_size, shape[1]*shape[2], shape[3]))
                smo_t = tf.add(smo, mod)
                return smo_t

            def while_condition(i, r, p, s, x): return tf.less(i, 50)

            def body(i, r, p, s, x):
                q = multi_sparse(p, self.A, shape[0], shape)
                alpha_t = tf.div(s, tf.matmul(tf.transpose(p, [0, 2, 1]),q))
                alpha_t = tf.map_fn(lambda x: tf.diag_part(x),alpha_t,dtype=tf.float32)
                alpha =tf.tile(tf.reshape(alpha_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                x = tf.add(x, tf.multiply(alpha, p))
                r = tf.subtract(r, tf.multiply(alpha, q))
                t = tf.matmul(tf.transpose(r, [0, 2, 1]),r)
                beta_t = tf.div(t, s)
                beta_t = tf.map_fn(lambda x: tf.diag_part(x),beta_t,dtype=tf.float32)
                beta = tf.tile(tf.reshape(beta_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                p = tf.add(r, tf.multiply(beta, p))
                s = t
                i = tf.add(i, 1)
                return i, r, p, s, x

            # zeros situation
            nn = tf.matmul(tf.transpose(modifier, [0, 2, 1]), modifier)
            oo = tf.zeros_like(nn)
            noeq = tf.equal(nn, oo)
            noeq_int = tf.to_int32(noeq)
            noeq_res = tf.equal(tf.reduce_sum(noeq_int), tf.reduce_sum(tf.ones_like(noeq_int)))

            def f_false(modifier):
                i = tf.constant(0)
                r = modifier - multi_sparse(modifier, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = modifier
                i, r, p, s, smo_mod = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])
                ## norm
                z_d = tf.ones_like(modifier)
                i = tf.constant(0)
                r = z_d - multi_sparse(z_d, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = z_d
                i, r, p, s, div_z = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])

                smo_mod = tf.div(smo_mod, div_z)
                smo_mod = tf.reshape(smo_mod, shape)
                return smo_mod

            def f_true(mm):
                smo_mod = tf.reshape(mm,shape)
                return smo_mod

            smo_mod = tf.cond(noeq_res, lambda: f_true(modifier),lambda: f_false(modifier))

            self.modifier =tf.reshape(modifier,shape)
            eta = smo_mod



        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True


class SmoothBasicIterativeMethodCG(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(SmoothBasicIterativeMethodCG, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, Aa, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        A = Aa
        self.A=A
        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate(x + eta, **fgm_params) - x

            shape = eta.shape
            modifier = tf.reshape(eta,(shape[0],shape[1]*shape[2],shape[3]))
            def multi_sparse(mod, A, batch_size, shape):
                im_mod = tf.reshape(
                    mod, (batch_size, shape[1], shape[2],shape[3]))
                im_mod = tf.pad(im_mod, paddings=[[0, 0], [1, 1], [
                                1, 1], [0, 0]], mode="CONSTANT")
                im_r = tf.slice(im_mod, [0, 0, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_l = tf.slice(im_mod, [0, 2, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_u = tf.slice(im_mod, [0, 1, 0, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_d = tf.slice(im_mod, [0, 1, 2, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_r = tf.reshape(im_r, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_l = tf.reshape(im_l, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_u = tf.reshape(im_u, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_d = tf.reshape(im_d, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_a = tf.concat([im_r, im_l, im_u, im_d], 1)
                A = tf.reshape(A, (batch_size, 4, shape[1]*shape[2], shape[3]))
                smo = tf.multiply(A, im_a)
                smo = tf.reduce_sum(smo, axis=1)
                smo = tf.reshape(smo, (batch_size, shape[1]*shape[2], shape[3]))
                smo_t = tf.add(smo, mod)
                return smo_t

            def while_condition(i, r, p, s, x): return tf.less(i, 50)

            def body(i, r, p, s, x):
                q = multi_sparse(p, self.A, shape[0], shape)
                alpha_t = tf.div(s, tf.matmul(tf.transpose(p, [0, 2, 1]),q))
                alpha_t = tf.map_fn(lambda x: tf.diag_part(x),alpha_t,dtype=tf.float32)
                alpha =tf.tile(tf.reshape(alpha_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                x = tf.add(x, tf.multiply(alpha, p))
                r = tf.subtract(r, tf.multiply(alpha, q))
                t = tf.matmul(tf.transpose(r, [0, 2, 1]),r)
                beta_t = tf.div(t, s)
                beta_t = tf.map_fn(lambda x: tf.diag_part(x),beta_t,dtype=tf.float32)
                beta = tf.tile(tf.reshape(beta_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                p = tf.add(r, tf.multiply(beta, p))
                s = t
                i = tf.add(i, 1)
                return i, r, p, s, x

            # zeros situation
            nn = tf.matmul(tf.transpose(modifier, [0, 2, 1]), modifier)
            oo = tf.zeros_like(nn)
            noeq = tf.equal(nn, oo)
            noeq_int = tf.to_int32(noeq)
            noeq_res = tf.equal(tf.reduce_sum(noeq_int), tf.reduce_sum(tf.ones_like(noeq_int)))

            def f_false(modifier):
                i = tf.constant(0)
                r = modifier - multi_sparse(modifier, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = modifier
                i, r, p, s, smo_mod = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])
                ## norm
                z_d = tf.ones_like(modifier)
                i = tf.constant(0)
                r = z_d - multi_sparse(z_d, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = z_d
                i, r, p, s, div_z = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])

                smo_mod = tf.div(smo_mod, div_z)
                smo_mod = tf.reshape(smo_mod, shape)
                return smo_mod

            def f_true(mm):
                smo_mod = tf.reshape(mm,shape)
                return smo_mod

            smo_mod = tf.cond(noeq_res, lambda: f_true(modifier),lambda: f_false(modifier))

            self.modifier =tf.reshape(modifier,shape)
            eta = smo_mod

            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm


        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True



class DoubleSmoothBasicIterativeMethodCG(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(DoubleSmoothBasicIterativeMethodCG, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, Aa, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        A = Aa
        self.A=A
        # try
        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate( x + eta, **fgm_params) - x
            #eta = FGM.generate(x + eta, **fgm_params) - x

            shape = eta.shape
            modifier = tf.reshape(eta,(shape[0],shape[1]*shape[2],shape[3]))
            def multi_sparse(mod, A, batch_size, shape):
                im_mod = tf.reshape(
                    mod, (batch_size, shape[1], shape[2],shape[3]))
                im_mod = tf.pad(im_mod, paddings=[[0, 0], [1, 1], [
                                1, 1], [0, 0]], mode="CONSTANT")
                im_r = tf.slice(im_mod, [0, 0, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_l = tf.slice(im_mod, [0, 2, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_u = tf.slice(im_mod, [0, 1, 0, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_d = tf.slice(im_mod, [0, 1, 2, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_r = tf.reshape(im_r, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_l = tf.reshape(im_l, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_u = tf.reshape(im_u, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_d = tf.reshape(im_d, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_a = tf.concat([im_r, im_l, im_u, im_d], 1)
                A = tf.reshape(A, (batch_size, 4, shape[1]*shape[2], shape[3]))
                smo = tf.multiply(A, im_a)
                smo = tf.reduce_sum(smo, axis=1)
                smo = tf.reshape(smo, (batch_size, shape[1]*shape[2], shape[3]))
                smo_t = tf.add(smo, mod)
                return smo_t

            def while_condition(i, r, p, s, x): return tf.less(i, 50)

            def body(i, r, p, s, x):
                q = multi_sparse(p, self.A, shape[0], shape)
                alpha_t = tf.div(s, tf.matmul(tf.transpose(p, [0, 2, 1]),q))
                alpha_t = tf.map_fn(lambda x: tf.diag_part(x),alpha_t,dtype=tf.float32)
                alpha =tf.tile(tf.reshape(alpha_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                x = tf.add(x, tf.multiply(alpha, p))
                r = tf.subtract(r, tf.multiply(alpha, q))
                t = tf.matmul(tf.transpose(r, [0, 2, 1]),r)
                beta_t = tf.div(t, s)
                beta_t = tf.map_fn(lambda x: tf.diag_part(x),beta_t,dtype=tf.float32)
                beta = tf.tile(tf.reshape(beta_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                p = tf.add(r, tf.multiply(beta, p))
                s = t
                i = tf.add(i, 1)
                return i, r, p, s, x

            # zeros situation
            nn = tf.matmul(tf.transpose(modifier, [0, 2, 1]), modifier)
            oo = tf.zeros_like(nn)
            noeq = tf.equal(nn, oo)
            noeq_int = tf.to_int32(noeq)
            noeq_res = tf.equal(tf.reduce_sum(noeq_int), tf.reduce_sum(tf.ones_like(noeq_int)))

            def f_false(modifier):
                i = tf.constant(0)
                r = modifier - multi_sparse(modifier, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = modifier
                i, r, p, s, smo_mod = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])
                ## norm
                z_d = tf.ones_like(modifier)
                i = tf.constant(0)
                r = z_d - multi_sparse(z_d, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = z_d
                i, r, p, s, div_z = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])

                smo_mod = tf.div(smo_mod, div_z)
                smo_mod = tf.reshape(smo_mod, shape)
                return smo_mod

            def f_true(mm):
                smo_mod = tf.reshape(mm,shape)
                return smo_mod

            smo_mod = tf.cond(noeq_res, lambda: f_true(modifier),lambda: f_false(modifier))

            self.modifier =tf.reshape(modifier,shape)
            eta = smo_mod

            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm

            modifier = tf.reshape(eta,(shape[0],shape[1]*shape[2],shape[3]))
            eta = tf.cond(noeq_res, lambda: f_true(modifier),lambda: f_false(modifier))


        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

class SmoothOurIterativeMethodCG(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(SmoothOurIterativeMethodCG, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, Aa,y_ph, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from . import utils_tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        x_shape = x.shape
        eta = tf.zeros(x_shape)

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps':1 , y_kwarg: y, 'ord': np.inf,
              'clip_min': self.clip_min, 'clip_max': self.clip_max}

        A = Aa
        self.A=A
        adv_x = x
        def while_condition_stop(eta,adv_x):
            logits = self.model.get_logits(x)
            acc=tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits,y_ph,1),tf.float32))
            cond1=tf.greater_equal(acc,1)
            reduc_ind = list(xrange(1, len(eta.get_shape())))
            norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                        reduction_indices=reduc_ind,
                        keep_dims=True))
            cond2_sum=tf.less(norm,tf.ones_like(norm)*self.eps)
            cond2 = tf.reduce_any(cond2_sum)
            return tf.logical_and(cond1,cond2)

        def big_body(eta,adv_x):
            # Compute this step's perturbation
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate(adv_x, **fgm_params) - adv_x

            shape = eta.shape
            modifier = tf.reshape(eta,(shape[0],shape[1]*shape[2],shape[3]))
            def multi_sparse(mod, A, batch_size, shape):
                im_mod = tf.reshape(
                    mod, (batch_size, shape[1], shape[2],shape[3]))
                im_mod = tf.pad(im_mod, paddings=[[0, 0], [1, 1], [
                                1, 1], [0, 0]], mode="CONSTANT")
                im_r = tf.slice(im_mod, [0, 0, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_l = tf.slice(im_mod, [0, 2, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_u = tf.slice(im_mod, [0, 1, 0, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_d = tf.slice(im_mod, [0, 1, 2, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_r = tf.reshape(im_r, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_l = tf.reshape(im_l, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_u = tf.reshape(im_u, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_d = tf.reshape(im_d, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_a = tf.concat([im_r, im_l, im_u, im_d], 1)
                A = tf.reshape(A, (batch_size, 4, shape[1]*shape[2], shape[3]))
                smo = tf.multiply(A, im_a)
                smo = tf.reduce_sum(smo, axis=1)
                smo = tf.reshape(smo, (batch_size, shape[1]*shape[2], shape[3]))
                smo_t = tf.add(smo, mod)
                return smo_t

            def while_condition(i, r, p, s, x): return tf.less(i, 50)

            def body(i, r, p, s, x):
                q = multi_sparse(p, self.A, shape[0], shape)
                alpha_t = tf.div(s, tf.matmul(tf.transpose(p, [0, 2, 1]),q))
                alpha_t = tf.map_fn(lambda x: tf.diag_part(x),alpha_t,dtype=tf.float32)
                alpha =tf.tile(tf.reshape(alpha_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                x = tf.add(x, tf.multiply(alpha, p))
                r = tf.subtract(r, tf.multiply(alpha, q))
                t = tf.matmul(tf.transpose(r, [0, 2, 1]),r)
                beta_t = tf.div(t, s)
                beta_t = tf.map_fn(lambda x: tf.diag_part(x),beta_t,dtype=tf.float32)
                beta = tf.tile(tf.reshape(beta_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                p = tf.add(r, tf.multiply(beta, p))
                s = t
                i = tf.add(i, 1)
                return i, r, p, s, x

            # zeros situation
            nn = tf.matmul(tf.transpose(modifier, [0, 2, 1]), modifier)
            oo = tf.zeros_like(nn)
            noeq = tf.equal(nn, oo)
            noeq_int = tf.to_int32(noeq)
            noeq_res = tf.equal(tf.reduce_sum(noeq_int), tf.reduce_sum(tf.ones_like(noeq_int)))

            def f_false(modifier):
                i = tf.constant(0)
                r = modifier - multi_sparse(modifier, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = modifier
                i, r, p, s, smo_mod = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])
                ## norm
                z_d = tf.ones_like(modifier)
                i = tf.constant(0)
                r = z_d - multi_sparse(z_d, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = z_d
                i, r, p, s, div_z = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])

                smo_mod = tf.div(smo_mod, div_z)
                smo_mod = tf.reshape(smo_mod, shape)
                return smo_mod

            def f_true(mm):
                smo_mod = tf.reshape(mm,shape)
                return smo_mod

            smo_mod = tf.cond(noeq_res, lambda: f_true(modifier),lambda: f_false(modifier))

            self.modifier =tf.reshape(modifier,shape)
            eta = smo_mod

            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps_iter, self.eps_iter)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps_iter / norm
                adv_x = adv_x + eta
                if self.clip_min is not None and self.clip_max is not None:
                    adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
            return eta,adv_x

        eta,adv_x = tf.while_loop(while_condition_stop,big_body,[eta,adv_x])

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

class SmoothMomentumIterativeMethod(Attack):

    """
    The Momentum Iterative Method (Dong et al. 2017). This method won
    the first places in NIPS 2017 Non-targeted Adversarial Attacks and
    Targeted Adversarial Attacks. The original paper used hard labels
    for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a MomentumIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(SmoothMomentumIterativeMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32,
                                'A':np.float32}
        self.structural_kwargs = ['ord', 'nb_iter', 'decay_factor']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param decay_factor: (optional) Decay factor for the momentum term.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        momentum = 0
        adv_x = x

        # Fix labels to the first model predictions for loss computation
        y, nb_classes = self.get_or_guess_labels(x, kwargs)
        y = y / tf.reduce_sum(y, 1, keep_dims=True)
        targeted = (self.y_target is not None)

        from . import utils_tf
        for i in range(self.nb_iter):
            # Compute loss
            preds = self.model.get_probs(adv_x)
            loss = utils_tf.model_loss(y, preds, mean=False)
            if targeted:
                loss = -loss

            # Define gradient of loss wrt input
            grad, = tf.gradients(loss, adv_x)

            # Normalize current gradient and add it to the accumulated gradient
            red_ind = list(xrange(1, len(grad.get_shape())))
            avoid_zero_div = 1e-12
            grad = grad / tf.maximum(avoid_zero_div,
                                     tf.reduce_mean(tf.abs(grad),
                                                    red_ind,
                                                    keep_dims=True))
            momentum = self.decay_factor * momentum + grad

            if self.ord == np.inf:
                normalized_grad = tf.sign(momentum)
            elif self.ord == 1:
                norm = tf.maximum(avoid_zero_div,
                                  tf.reduce_sum(tf.abs(momentum),
                                                red_ind,
                                                keep_dims=True))
                normalized_grad = momentum / norm
            elif self.ord == 2:
                square = tf.reduce_sum(tf.square(momentum),
                                       red_ind,
                                       keep_dims=True)
                norm = tf.sqrt(tf.maximum(avoid_zero_div, square))
                normalized_grad = momentum / norm
            else:
                raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                          "currently implemented.")

            # Update and clip adversarial example in current iteration
            scaled_grad = self.eps_iter * normalized_grad
            adv_x = adv_x + scaled_grad
            adv_x = x + utils_tf.clip_eta(adv_x - x, self.ord, self.eps)

            if self.clip_min is not None and self.clip_max is not None:
                adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

            adv_x = tf.stop_gradient(adv_x)
            adv   = tf.reshape(adv_x,[-1,1,784])
            adv   = tf.matmul(adv,self.A)
            adv   = tf.matmul(adv,self.At)
            adv_x = tf.reshape(adv,[-1,28,28,1])

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.06, nb_iter=10, y=None,
                     ord=np.inf, decay_factor=1.0,
                     clip_min=None, clip_max=None,
                     y_target=None,A=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param decay_factor: (optional) Decay factor for the momentum term.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.decay_factor = decay_factor
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.A = A['pi']
        self.At = A['pit']

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

class SPSA(Attack):
    """
    This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666
    (Uesato et al. 2018). SPSA is a gradient-free optimization method, which
    is useful when the model is non-differentiable, or more generally, the
    gradients do not point in useful directions.
    """

    def __init__(self, model, back='tf', sess=None):
        super(SPSA, self).__init__(model, back, sess)
        assert isinstance(self.model, Model)

    def generate(self, x, y=None, y_target=None, epsilon=None, num_steps=None,
                 is_targeted=False, early_stop_loss_threshold=None,
                 learning_rate=0.01, delta=0.01, batch_size=128, spsa_iters=1,
                 is_debug=False):
        """
        Generate symbolic graph for adversarial examples.
        :param x: The model's symbolic inputs. Must be a batch of size 1.
        :param y: A Tensor or None. The index of the correct label.
        :param y_target: A Tensor or None. The index of the target label in a
                         targeted attack.
        :param epsilon: The size of the maximum perturbation, measured in the
                        L-infinity norm.
        :param num_steps: The number of optimization steps.
        :param is_targeted: Whether to use a targeted or untargeted attack.
        :param early_stop_loss_threshold: A float or None. If specified, the
                                          attack will end as soon as the loss
                                          is below `early_stop_loss_threshold`.
        :param learning_rate: Learning rate of ADAM optimizer.
        :param delta: Perturbation size used for SPSA approximation.
        :param batch_size: Number of inputs to evaluate at a single time. Note
                           that the true batch size (the number of evaluated
                           inputs for each update) is `batch_size * spsa_iters`
        :param spsa_iters: Number of model evaluations before performing an
                           update, where each evaluation is on `batch_size`
                           different inputs.
        :param is_debug: If True, print the adversarial loss after each update.
        """
        from .attacks_tf_hw import SPSAAdam, pgd_attack, margin_logit_loss

        optimizer = SPSAAdam(lr=learning_rate, delta=delta,
                             num_samples=batch_size, num_iters=spsa_iters)

        def loss_fn(x, label):
            logits = self.model.get_logits(x)
            loss_multiplier = 1 if is_targeted else -1
            return loss_multiplier * margin_logit_loss(
                logits, label, num_classes=self.model.num_classes)

        y_attack = y_target if is_targeted else y
        adv_x = pgd_attack(
            loss_fn, x, y_attack, epsilon, num_steps=num_steps,
            optimizer=optimizer,
            early_stop_loss_threshold=early_stop_loss_threshold,
            is_debug=is_debug,
        )
        return adv_x

class CarliniWagnerL2_my(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(CarliniWagnerL2_my, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import CarliniWagnerL2_my as CWL2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CWL2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap = tf.py_func(cw_wrap, [x, labels], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

class Clip_origin(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_origin, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Clip_origin as CWL2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CWL2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap = tf.py_func(cw_wrap, [x, labels], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

class Clip_version_old(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_version_old, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32,
                                'A':tf.float32,
                                'At':tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Clip_version_old as CLV2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        

        At = kwargs['A']
        A = kwargs['At']
        
        attack = CLV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cv_wrap(x_val, y_val, A_val, At_val):
            return np.array(attack.attack(x_val, y_val, A_val, At_val), dtype=np.float32)
        wrap = tf.py_func(cv_wrap, [x, labels, A, At], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1,A=None,At=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

class Clip_version_debug(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_version_debug, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32,
                                'A':tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from attacks_tf import Clip_version_debug as CLV2
        self.parse_params(**kwargs)

        A = kwargs['A']
        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CLV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cv_wrap(x_val, y_val,A_val):
            a,b = attack.attack(x_val,y_val,A_val)
            return np.array(a, dtype=np.float32), np.array(b,dtype=np.float32)
        wrap,b = tf.py_func(cv_wrap, [x, labels, A], (tf.float32, tf.float32))

        return wrap,b

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1,A=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

class Clip_version_eig_norm(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_version_eig_norm, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32,
                                'A':tf.float32,
                                'At':tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max','eig_num']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param eig_num: number of the eig value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Clip_version_eig_norm as CLV2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        

        At = kwargs['At']
        A = kwargs['A']
        
        attack = CLV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max, self.eig_num,
                      nb_classes, x.get_shape().as_list()[1:])

        def cv_wrap(x_val, y_val, A_val, At_val):
            return np.array(attack.attack(x_val, y_val, A_val, At_val), dtype=np.float32)
        wrap = tf.py_func(cv_wrap, [x, labels, A, At], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1, eig_num=50, A=None,At=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eig_num = eig_num

class Clip_version_eig(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_version_eig, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32,
                                'A':tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max','eig_num']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param eig_num: number of the eig value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Clip_version_eig as CLV2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        

        A = kwargs['A']
        
        attack = CLV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max, self.eig_num,
                      nb_classes, x.get_shape().as_list()[1:])

        def cv_wrap(x_val, y_val, A_val):
            return np.array(attack.attack(x_val, y_val, A_val), dtype=np.float32)
        wrap = tf.py_func(cv_wrap, [x, labels, A], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1, eig_num=50, A=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eig_num = eig_num

##############################
# iterative version
#############################

class Clip_version_debug_iter(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_version_debug_iter, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32,
                                'A':tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Clip_version_debug_iter as CLV2
        self.parse_params(**kwargs)

        A = kwargs['A']
        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CLV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cv_wrap(x_val, y_val,A_val):
            r,l1,l2 = attack.attack(x_val, y_val,A_val)
            return np.array(r,dtype=np.float32), np.array(l1,dtype=np.float32),np.array(l2,dtype=np.float32)
            #return np.array(attack.attack(x_val, y_val,A_val), dtype=np.float32)
        #wrap = tf.py_func(cv_wrap, [x, labels, A], tf.float32)
        wrap, l1, l2 = tf.py_func(cv_wrap, [x, labels, A], [tf.float32,tf.float32,tf.float32])
        return wrap, l1, l2


    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1,A=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

class CarliniWagnerL2_iter(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(CarliniWagnerL2_iter, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import CarliniWagnerL2_iter as CWL2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CWL2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cv_wrap(x_val, y_val):
            r,l1,l2 = attack.attack(x_val, y_val)
            return np.array(r,dtype=np.float32), np.array(l1,dtype=np.float32),np.array(l2,dtype=np.float32)

        wrap, l1, l2 = tf.py_func(cv_wrap, [x, labels], [tf.float32,tf.float32,tf.float32])
        return wrap, l1, l2

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

###################
# current imporve version
#################

class Clip_version_debug_speed(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_version_debug_speed, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32,
                                'A':tf.float32,
                                'div_z':tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Clip_version_debug_speed as CLV2
        self.parse_params(**kwargs)

        A = kwargs['A']
        div_z = kwargs['div_z']
        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CLV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cv_wrap(x_val, y_val,A_val,d_val):
            return np.array(attack.attack(x_val, y_val,A_val,d_val), dtype=np.float32)
        wrap = tf.py_func(cv_wrap, [x, labels, A, div_z], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1,A=None,div_z=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

#####################
# big image
#####################
class CarliniWagnerL2_big(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(CarliniWagnerL2_big, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import CarliniWagnerL2_big as CWL2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_label(x, kwargs)

        attack = CWL2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=np.float32)
        wrap = tf.py_func(cw_wrap, [x, labels], tf.float32)

        return wrap

    def get_or_guess_label(self, x, kwargs):
        import tensorflow as tf
        from inception_preprocessing import preprocess_for_eval

        if 'y' in kwargs and 'y_target' in kwargs:
            raise ValueError("Can not set both 'y' and 'y_target'.")
        elif 'y' in kwargs:
            labels = kwargs['y']
        elif 'y_target' in kwargs:
            labels = kwargs['y_target']
        else:
            shape = x.get_shape().as_list()[1:]
            x = tf.reshape(x,shape=shape)
            x = preprocess_for_eval(x,299,299)
            x = tf.reshape(x,[1,299,299,3])
            preds = self.model.get_probs(x)
            preds_max = tf.reduce_max(preds, 1, keep_dims=True)
            original_predictions = tf.to_float(tf.equal(preds,
                                                        preds_max))
            labels = tf.stop_gradient(original_predictions)
        if isinstance(labels, np.ndarray):
            nb_classes = labels.shape[1]
        else:
            nb_classes = labels.get_shape().as_list()[1]
        return labels, nb_classes

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max


class Clip_version_debug_big(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_version_debug_big, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32,
                                'A':tf.float32,
                                'div_z':tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Clip_version_debug_big as CLV2
        self.parse_params(**kwargs)

        A = kwargs['A']
        div_z = kwargs['div_z']
        labels, nb_classes = self.get_or_guess_label(x, kwargs)

        attack = CLV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cv_wrap(x_val, y_val,A_val,d_val):
            return np.array(attack.attack(x_val, y_val,A_val,d_val), dtype=np.float32)
        wrap = tf.py_func(cv_wrap, [x, labels, A, div_z], tf.float32)

        return wrap

    def get_or_guess_label(self, x, kwargs):
        import tensorflow as tf
        from inception_preprocessing import preprocess_for_eval

        if 'y' in kwargs and 'y_target' in kwargs:
            raise ValueError("Can not set both 'y' and 'y_target'.")
        elif 'y' in kwargs:
            labels = kwargs['y']
        elif 'y_target' in kwargs:
            labels = kwargs['y_target']
        else:
            shape = x.get_shape().as_list()[1:]
            x = tf.reshape(x,shape=shape)
            x = preprocess_for_eval(x,299,299)
            x = tf.reshape(x,[1,299,299,3])
            preds = self.model.get_probs(x)
            preds_max = tf.reduce_max(preds, 1, keep_dims=True)
            original_predictions = tf.to_float(tf.equal(preds,
                                                        preds_max))
            labels = tf.stop_gradient(original_predictions)
        if isinstance(labels, np.ndarray):
            nb_classes = labels.shape[1]
        else:
            nb_classes = labels.get_shape().as_list()[1]
        return labels, nb_classes

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1,A=None,div_z=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

class Clip_version_eig_imgnet(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(Clip_version_eig_imgnet, self).__init__(model, back, sess)

        import tensorflow as tf
        self.feedable_kwargs = {'y': tf.float32,
                                'y_target': tf.float32,
                                'A':tf.float32,
                                'At':tf.float32}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max','eig_num']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param eig_num: number of the eig value
        """
        import tensorflow as tf
        from .attacks_tf_hw import Clip_version_eig_imgnet as CLV2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        

        At = kwargs['At']
        A = kwargs['A']
        
        attack = CLV2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max, self.eig_num,
                      nb_classes, x.get_shape().as_list()[1:])

        def cv_wrap(x_val, y_val, A_val, At_val):
            return np.array(attack.attack(x_val, y_val, A_val, At_val), dtype=np.float32)
        wrap = tf.py_func(cv_wrap, [x, labels, A, At], tf.float32)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1, eig_num=50, A=None,At=None):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eig_num = eig_num

class SmoothBasicIterativeMethodCG(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        super(SmoothBasicIterativeMethodCG, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, Aa, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        A = Aa
        self.A=A
        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate(x + eta, **fgm_params) - x

            shape = eta.shape
            modifier = tf.reshape(eta,(shape[0],shape[1]*shape[2],shape[3]))
            def multi_sparse(mod, A, batch_size, shape):
                im_mod = tf.reshape(
                    mod, (batch_size, shape[1], shape[2],shape[3]))
                im_mod = tf.pad(im_mod, paddings=[[0, 0], [1, 1], [
                                1, 1], [0, 0]], mode="CONSTANT")
                im_r = tf.slice(im_mod, [0, 0, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_l = tf.slice(im_mod, [0, 2, 1, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_u = tf.slice(im_mod, [0, 1, 0, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_d = tf.slice(im_mod, [0, 1, 2, 0], [batch_size, shape[1], shape[2], shape[3]])
                im_r = tf.reshape(im_r, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_l = tf.reshape(im_l, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_u = tf.reshape(im_u, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_d = tf.reshape(im_d, (batch_size, 1, shape[1]*shape[2], shape[3]))
                im_a = tf.concat([im_r, im_l, im_u, im_d], 1)
                A = tf.reshape(A, (batch_size, 4, shape[1]*shape[2], shape[3]))
                smo = tf.multiply(A, im_a)
                smo = tf.reduce_sum(smo, axis=1)
                smo = tf.reshape(smo, (batch_size, shape[1]*shape[2], shape[3]))
                smo_t = tf.add(smo, mod)
                return smo_t

            def while_condition(i, r, p, s, x): return tf.less(i, 50)

            def body(i, r, p, s, x):
                q = multi_sparse(p, self.A, shape[0], shape)
                alpha_t = tf.div(s, tf.matmul(tf.transpose(p, [0, 2, 1]),q))
                alpha_t = tf.map_fn(lambda x: tf.diag_part(x),alpha_t,dtype=tf.float32)
                alpha =tf.tile(tf.reshape(alpha_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                x = tf.add(x, tf.multiply(alpha, p))
                r = tf.subtract(r, tf.multiply(alpha, q))
                t = tf.matmul(tf.transpose(r, [0, 2, 1]),r)
                beta_t = tf.div(t, s)
                beta_t = tf.map_fn(lambda x: tf.diag_part(x),beta_t,dtype=tf.float32)
                beta = tf.tile(tf.reshape(beta_t,[shape[0],1,shape[3]]),multiples=[1,shape[1]*shape[2],1])
                p = tf.add(r, tf.multiply(beta, p))
                s = t
                i = tf.add(i, 1)
                return i, r, p, s, x

            # zeros situation
            nn = tf.matmul(tf.transpose(modifier, [0, 2, 1]), modifier)
            oo = tf.zeros_like(nn)
            noeq = tf.equal(nn, oo)
            noeq_int = tf.to_int32(noeq)
            noeq_res = tf.equal(tf.reduce_sum(noeq_int), tf.reduce_sum(tf.ones_like(noeq_int)))

            def f_false(modifier):
                i = tf.constant(0)
                r = modifier - multi_sparse(modifier, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = modifier
                i, r, p, s, smo_mod = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])
                ## norm
                z_d = tf.ones_like(modifier)
                i = tf.constant(0)
                r = z_d - multi_sparse(z_d, self.A, shape[0], shape)
                p = r
                s = tf.matmul(tf.transpose(r, [0, 2, 1]), r)
                x = z_d
                i, r, p, s, div_z = tf.while_loop(
                    while_condition, body, [i, r, p, s, x])

                smo_mod = tf.div(smo_mod, div_z)
                smo_mod = tf.reshape(smo_mod, shape)
                return smo_mod

            def f_true(mm):
                smo_mod = tf.reshape(mm,shape)
                return smo_mod

            smo_mod = tf.cond(noeq_res, lambda: f_true(modifier),lambda: f_false(modifier))

            self.modifier =tf.reshape(modifier,shape)
            eta = smo_mod

            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm


        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True