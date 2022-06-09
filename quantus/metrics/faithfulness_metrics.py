"""This module contains the collection of faithfulness metrics to evaluate attribution-based explanations of neural network models."""
import itertools
import math
import random
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from .base import Metric, BatchedMetric
from ..helpers import asserts
from ..helpers import plotting
from ..helpers import utils
from ..helpers import perturb_func as perturb_funcs
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.normalise_func import normalise_by_negative
from ..helpers.similar_func import correlation_pearson, correlation_spearman
from ..helpers.perturb_func import baseline_replacement_by_indices
from ..helpers.perturb_func import baseline_replacement_by_patch
from ..typing import Patch


class FaithfulnessCorrelation(Metric):
    """
    Implementation of faithfulness correlation by Bhatt et al., 2020.

    The Faithfulness Correlation metric intend to capture an explanation's relative faithfulness
    (or 'fidelity') with respect to the model behaviour.

    Faithfulness correlation scores shows to what extent the predicted logits of each modified test point and
    the average explanation attribution for only the subset of features are (linearly) correlated, taking the
    average over multiple runs and test samples. The metric returns one float per input-attribution pair that
    ranges between -1 and 1, where higher scores are better.

    For each test sample, |S| features are randomly selected and replace them with baseline values (zero baseline
    or average of set). Thereafter, Pearson’s correlation coefficient between the predicted logits of each modified
    test point and the average explanation attribution for only the subset of features is calculated. Results is
    average over multiple runs and several test samples.

    References:
        1) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating feature-based model
        explanations." arXiv preprint arXiv:2005.00631 (2020).

    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            nr_runs (integer): The number of runs (for each input and explanation pair), default=100.
            subset_size (integer): The size of subset, default=224.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_pearson.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            return_aggregate (boolean): Indicates whether an aggregated(mean) metric is returned, default=True.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.nr_runs = self.kwargs.get("nr_runs", 100)
        self.subset_size = self.kwargs.get("subset_size", 224)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.return_aggregate = self.kwargs.get("return_aggregate", True)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', size of subset |S| 'subset_size'"
                    " and the number of runs (for each input and explanation pair) "
                    "'nr_runs'"
                ),
                citation=(
                    "Bhatt, Umang, Adrian Weller, and José MF Moura. 'Evaluating and aggregating "
                    "feature-based model explanations.' arXiv preprint arXiv:2005.00631 (2020)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = FaithfulnessCorrelation(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_value_smaller_than_input_size(
            x=x_batch_s, value=self.subset_size, value_name="subset_size"
        )

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(zip(x_batch_s, y_batch, a_batch), total=len(x_batch_s))

        for x, y, a in iterator:

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Predict on input.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(
                model.predict(x_input, softmax_act=False, **self.kwargs)[:, y]
            )

            logit_deltas = []
            att_sums = []

            # For each test data point, execute a couple of runs.
            for i_ix in range(self.nr_runs):

                # Randomly mask by subset size.
                a_ix = np.random.choice(a.shape[0], self.subset_size, replace=False)
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=a_ix,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(
                    model.predict(x_input, softmax_act=False, **self.kwargs)[:, y]
                )
                logit_deltas.append(float(y_pred - y_pred_perturb))

                # Sum attributions of the random subset.
                att_sums.append(np.sum(a[a_ix]))

            similarity = self.similarity_func(a=att_sums, b=logit_deltas)
            self.last_results.append(similarity)

        if self.return_aggregate:
            self.last_results = [np.mean(self.last_results)]
        else:
            self.last_results = self.last_results

        self.all_results.append(self.last_results)

        return self.last_results


class FaithfulnessEstimate(Metric):
    """
    Implementation of Faithfulness Estimate by Alvares-Melis at el., 2018a and 2018b.

    Computes the correlations of probability drops and the relevance scores on various points,
    showing the aggregate statistics.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            nr_runs (integer): The number of runs (for each input and explanation pair), default=100.
            subset_size (integer): The size of subset, default=224.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_spearman.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and similarity function "
                    "'similarity_func'"
                ),
                citation=(
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'Towards robust interpretability"
                    " with self-explaining neural networks.' arXiv preprint arXiv:1806.07538 (2018)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = FaithfulnessEstimate(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=x_batch_s.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=x_batch_s.shape[2:],
            )

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(zip(x_batch_s, y_batch, a_batch), total=len(x_batch_s))

        for x, y, a in iterator:

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Get indices of sorted attributions (descending).
            a_indices = np.argsort(-a)

            # Predict on input.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(
                model.predict(x_input, softmax_act=False, **self.kwargs)[:, y]
            )

            pred_deltas = []
            att_sums = []

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=a_ix,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(
                    model.predict(x_input, softmax_act=False, **self.kwargs)[:, y]
                )
                pred_deltas.append(float(y_pred - y_pred_perturb))

                # Sum attributions.
                att_sums.append(a[a_ix].sum())

            self.last_results.append(self.similarity_func(a=att_sums, b=pred_deltas))

        self.all_results.append(self.last_results)

        return self.last_results


class IterativeRemovalOfFeatures(Metric):
    """
    Implementation of IROF (Iterative Removal of Features) by Rieger at el., 2020.

    The metric computes the area over the curve per class for sorted mean importances
    of feature segments (superpixels) as they are iteratively removed (and prediction scores are collected),
    averaged over several test samples.

    References:
        1) Rieger, Laura, and Lars Kai Hansen. "Irof: a low resource evaluation metric for
        explanation methods." arXiv preprint arXiv:2003.08747 (2020).

    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            segmentation_method (string): Image segmentation method:'slic' or 'felzenszwalb', default="slic".
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="mean".
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.segmentation_method = self.kwargs.get("segmentation_method", "slic")
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "mean")
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the method to segment "
                    "the image 'segmentation_method' (including all its associated hyperparameters)"
                ),
                citation=(
                    "Rieger, Laura, and Lars Kai Hansen. 'Irof: a low resource evaluation metric "
                    "for explanation methods.' arXiv preprint arXiv:2003.08747 (2020)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = IROF(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        nr_channels = x_batch_s.shape[1]
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)), total=len(x_batch_s)
            )

        for ix, (x, y, a) in iterator:

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Predict on x.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(
                model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
            )

            # Segment image.
            segments = utils.get_superpixel_segments(
                img=np.moveaxis(x, 0, -1).astype("double"),
                segmentation_method=self.segmentation_method,
            )
            nr_segments = segments.max()
            asserts.assert_nr_segments(nr_segments=nr_segments)

            # Calculate average attribution of each segment.
            att_segs = np.zeros(nr_segments)
            for i, s in enumerate(range(nr_segments)):
                att_segs[i] = np.mean(a[:, segments == s])

            # Sort segments based on the mean attribution (descending order).
            s_indices = np.argsort(-att_segs)

            preds = []

            for i_ix, s_ix in enumerate(s_indices):

                # Perturb input by indices of attributions.
                a_ix = np.nonzero(np.repeat((segments == s_ix).flatten(), nr_channels))[
                    0
                ]

                x_perturbed = self.perturb_func(
                    arr=x_input.flatten(),
                    indices=a_ix,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(
                    model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
                )
                # Normalise the scores to be within [0, 1].
                preds.append(float(y_pred_perturb / y_pred))

            # self.last_results.append(1-auc(preds, np.arange(0, len(preds))))
            self.last_results.append(np.trapz(np.array(preds), dx=1.0))

        self.last_results = [np.mean(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def aggregated_score(self):
        """Calculate the area over the curve (AOC) score for several test samples."""
        return [np.mean(results) for results in self.all_results]


class MonotonicityArya(Metric):
    """
    Implementation of Montonicity Metric by Arya at el., 2019.

    Montonicity tests if adding more positive evidence increases the probability
    of classification in the specified class.

    It captures attributions' faithfulness by incrementally adding each attribute
    in order of increasing importance and evaluating the effect on model performance.
    As more features are added, the performance of the model is expected to increase
    and thus result in monotonically increasing model performance.

    References:
        1) Arya, Vijay, et al. "One explanation does not fit all: A toolkit and taxonomy of ai explainability
        techniques." arXiv preprint arXiv:1909.03012 (2019).
        2) Luss, Ronny, et al. "Generating contrastive explanations with monotonic attribute functions."
        arXiv preprint arXiv:1905.12698 (2019).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            concept_influence (boolean): Indicates whether concept influence metric is used.
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("baseline value 'perturb_baseline'"),
                citation=(
                    "Arya, Vijay, et al. 'One explanation does not fit all: A toolkit and taxonomy"
                    " of ai explainability techniques.' arXiv preprint arXiv:1909.03012 (2019)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[bool]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a list of bool(s) with the evaluation outcome of concerned batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = MonotonicityArya(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=x_batch_s.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=x_batch_s.shape[2:],
            )

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(zip(x_batch_s, y_batch, a_batch), total=len(x_batch_s))

        for x, y, a in iterator:

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Get indices of sorted attributions (ascending).
            a_indices = np.argsort(a)

            preds = []

            # Copy the input x but fill with baseline values.
            baseline_value = utils.get_baseline_value(
                choice=self.perturb_baseline, arr=x
            )
            x_baseline = np.full(x.shape, baseline_value).flatten()

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_baseline = self.perturb_func(
                    arr=x_baseline,
                    indices=a_ix,
                    **self.kwargs,
                )

                # Predict on perturbed input x (that was initially filled with a constant 'perturb_baseline' value).
                x_input = model.shape_input(x_baseline, x.shape, channel_first=True)
                y_pred_perturb = float(
                    model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
                )
                preds.append(y_pred_perturb)

            self.last_results.append(np.all(np.diff(preds) >= 0))

        self.all_results.append(self.last_results)

        return self.last_results


class MonotonicityNguyen(Metric):
    """
    Implementation of Montonicity Metric by Nguyen at el., 2020.

    Monotonicity measures the (Spearman’s) correlation coefficient of the absolute values of the attributions
    and the uncertainty in probability estimation. The paper argues that if attributions are not monotonic
    then they are not providing the correct importance of the feature.

    References:
        1) Nguyen, An-phi, and María Rodríguez Martínez. "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform".
            eps (float): Attributions threshold, default=1e-5.
            nr_samples (integer): The number of samples to iterate over, default=100.
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_spearman.
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.eps = self.kwargs.get("eps", 1e-5)
        self.nr_samples = self.kwargs.get("nr_samples", 100)
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', threshold value 'eps' and number "
                    "of samples to iterate over 'nr_samples'"
                ),
                citation=(
                    "Nguyen, An-phi, and María Rodríguez Martínez. 'On quantitative aspects of "
                    "model interpretability.' arXiv preprint arXiv:2007.07584 (2020)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = MonotonicityNguyen(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=x_batch_s.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=x_batch_s.shape[2:],
            )

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(zip(x_batch_s, y_batch, a_batch), total=len(x_batch_s))

        for x, y, a in iterator:

            # Predict on input x.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(
                model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
            )

            inv_pred = 1.0 if np.abs(y_pred) < self.eps else 1.0 / np.abs(y_pred)
            inv_pred = inv_pred**2

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Get indices of sorted attributions (ascending).
            a_indices = np.argsort(a)

            atts = []
            vars = []

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]

                y_pred_perturbs = []

                for n in range(self.nr_samples):

                    x_perturbed = self.perturb_func(
                        arr=x,
                        indices=a_ix,
                        **self.kwargs,
                    )
                    asserts.assert_perturbation_caused_change(
                        x=x, x_perturbed=x_perturbed
                    )

                    # Predict on perturbed input x.
                    x_input = model.shape_input(
                        x_perturbed, x.shape, channel_first=True
                    )
                    y_pred_perturb = float(
                        model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
                    )
                    y_pred_perturbs.append(y_pred_perturb)

                vars.append(
                    float(
                        np.mean((np.array(y_pred_perturbs) - np.array(y_pred)) ** 2)
                        * inv_pred
                    )
                )
                atts.append(float(sum(a[a_ix])))

            self.last_results.append(self.similarity_func(a=atts, b=vars))

        self.all_results.append(self.last_results)

        return self.last_results


class PixelFlipping(BatchedMetric):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            perturb_baseline: Any = "black",
            order: str = 'morf',
            features_in_step: int = 1,
            max_steps_per_input: Optional[int] = None,
            prediction_difference: bool = False,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
            prediction_difference (bool): If true, scores will represent difference to
                                          unperturbed output on target class, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative
        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        if plot_func is None:
            plot_func = plotting.plot_pixel_flipping_experiment

        # TODO: deprecate perturb_baseline keyword and use perturb_kwargs exclusively in later versions
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs = {
            'perturb_baseline': perturb_baseline,
            **perturb_func_kwargs,
        }

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        self.features_in_step = features_in_step
        self.max_steps_per_input = max_steps_per_input
        self.order = order.lower()
        self.prediction_difference = prediction_difference

        # Asserts and warnings.
        asserts.assert_attributions_order(order=self.order)
        self.disable_warnings = disable_warnings
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("baseline value 'perturb_baseline'"),
                citation=(
                    "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                    " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                    "e0130140"
                ),
            )

    def __call__(
            self,
            model: ModelInterface,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.array] = None,
            batch_size: int = 64,
            **kwargs,
    ) -> List[List[float]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = PixelFlipping(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            batch_size=batch_size,
            **kwargs,
        )

    def process_batch(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: np.ndarray,
            a_batch: np.ndarray,
            s_batch: Optional[np.ndarray] = None,
            perturb_func: Callable = None,
            perturb_func_kwargs: Optional[Dict] = None,
    ):
        # TODO: use check function for this, but maybe already in __call__
        if perturb_func is None:
            raise ValueError("perturb_func must not be None")
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        # create array for scores of each perturbation step
        batch_size = x_batch.shape[0]
        n_steps = math.ceil(a_batch[0].size / self.features_in_step)
        preds = [[None for _ in range(n_steps)] for _ in range(batch_size)]

        # create generator for perturbed batches
        perturbation_generator = self.generate_perturbed_batches(
            x_batch=x_batch,
            a_batch=a_batch,
            n_steps=n_steps,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
        )

        if self.prediction_difference:
            # Predict on original unperturbed input x.
            x_input = model.shape_input(
                x=x_batch,
                shape=x_batch.shape,
                channel_first=True,
                batched=True,
            )
            y_batch_pred = model.predict(x=x_input, softmax_act=True)

        for step, x_batch_perturbed in enumerate(perturbation_generator):

            # Predict on perturbed input x.
            x_input = model.shape_input(
                x=x_batch_perturbed,
                shape=x_batch.shape,
                channel_first=True,
                batched=True,
            )

            y_batch_pred_perturb = model.predict(x=x_input, softmax_act=True)

            # TODO: get rid of for loop
            iterator = zip(y_batch, y_batch_pred_perturb)
            for ix, (y, y_pred_perturb) in enumerate(iterator):
                y_pred_perturb_target = float(y_pred_perturb[y])

                # Use difference to prediction on unperturbed input as score.
                if self.prediction_difference:
                    y_pred_target = float(y_batch_pred[ix, y])
                    preds[ix][step] = y_pred_target - y_pred_perturb_target

                # Use prediction on perturbed input as score.
                else:
                    preds[ix][step] = y_pred_perturb_target

        return preds

    def generate_perturbed_batches(
            self,
            x_batch: np.ndarray,
            a_batch: np.ndarray,
            n_steps: int,
            perturb_func: Callable,
            perturb_func_kwargs: Dict[str, Any],
    ) -> Iterator[np.ndarray]:

        a_batch_indices = self.get_sorted_attribution_indices(
            x_batch=x_batch, a_batch=a_batch, order=self.order,
        )

        x_batch_perturbed = x_batch.copy()
        for step in range(n_steps):

            # Perturb input by indices of attributions.
            perturb_start_ix = self.features_in_step * step
            perturb_end_ix = self.features_in_step * (step + 1)
            perturb_indices = a_batch_indices[:, perturb_start_ix:perturb_end_ix]

            # Inplace perturbation.
            perturb_funcs.perturb_batch_on_indices(
                arr=x_batch_perturbed,
                indices=perturb_indices,
                perturb_func=perturb_func,
                **perturb_func_kwargs,
            )

            # Check for changes in perturbation.
            for x, x_perturbed in zip(x_batch, x_batch_perturbed):
                asserts.assert_perturbation_caused_change(
                    x=x, x_perturbed=x_perturbed)

            yield x_batch_perturbed

    def get_sorted_attribution_indices(
            self,
            x_batch: np.ndarray,
            a_batch: np.ndarray,
            order: str,
    ) -> np.ndarray:

        # Get indices of sorted attributions (descending).
        batch_size = x_batch.shape[0]
        flat_shape = a_batch[0].flatten().shape
        a_batch_flat = np.zeros((batch_size, *flat_shape))
        a_batch_indices = np.zeros((batch_size, *flat_shape), dtype=np.int)

        # generate ordered indices for complete batch
        # TODO: do this in a more readable way or even better: without for-loop
        for i in range(batch_size):
            a_batch_flat[i] = a_batch[i].flatten()

            if order.lower() == "random":
                # Order attributions randomly.
                n_indices = len(a_batch_flat[i])
                a_batch_indices[i] = random.sample(list(range(n_indices)),
                                                   k=n_indices)

            elif order.lower() == "morf":
                # Order attributions according to the most relevant first.
                a_batch_indices[i] = np.argsort(a_batch_flat[i])[::-1]

            elif order.lower() == 'lerf':
                # Order attributions according to the least relevant first.
                a_batch_indices[i] = np.argsort(a_batch_flat[i])

        return a_batch_indices
        

    def preprocess(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            S: np.ndarray,
            model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=X.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=X.shape[2:],
            )
        return X, Y, A, S, model


class RegionPerturbation(BatchedMetric):
    """

    Implementation of Region Perturbation by Samek et al., 2015.

    Consider a greedy iterative procedure that consists of measuring how the class
    encoded in the image (e.g. as measured by the function f) disappears when we
    progressively remove information from the image x, a process referred to as
    region perturbation, at the specified locations.

    References:
        1) Samek, Wojciech, et al. "Evaluating the visualization of what a deep
        neural network has learned." IEEE transactions on neural networks and
        learning systems 28.11 (2016): 2660-2673.

    Current assumptions:
        -Done according to Most Relevant First (MoRF) and Area Over the Perturbation Curve
        (AOPC).
        - 9 x 9 patch sizes was used in the paper as regions, but using 8 x 8
        to make sure non-overlapping
        - they called it "area over the MoRF perturbation curve" it
        looks like a simple deduction of function outputs?

    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            perturb_baseline: Any = "uniform",
            order: str = 'morf',
            regions_evaluation: int = 100,
            patch_size: int = 8,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_patch.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform".
            regions_evaluation (integer): The number of regions to evaluate, default=100.
            patch_size (integer): The patch size for masking, default=8.
            order (string): Indicates whether attributions are ordered randomly ("random"),
            according to the most relevant first ("MoRF"), or least relevant first, default="MoRF".
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative
        if perturb_func is None:
            perturb_func = baseline_replacement_by_patch
        if plot_func is None:
            plot_func = plotting.plot_region_perturbation_experiment

        # TODO: deprecate perturb_baseline keyword and use perturb_kwargs exclusively in later versions
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs = {
            'perturb_baseline': perturb_baseline,
            **perturb_func_kwargs,
        }

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        self.regions_evaluation = regions_evaluation
        self.patch_size = patch_size
        self.order = order.lower()

        # Asserts and warnings.
        asserts.assert_attributions_order(order=self.order)
        self.disable_warnings = disable_warnings
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline'"
                    ", the patch size for masking 'patch_size'"
                    " and number of regions to evaluate 'regions_evaluation'"
                ),
                citation=(
                    "Samek, Wojciech, et al. 'Evaluating the visualization of what a deep"
                    " neural network has learned.' IEEE transactions on neural networks and"
                    " learning systems 28.11 (2016): 2660-2673"
                ),
            )

    def __call__(
            self,
            model: ModelInterface,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.array] = None,
            batch_size: int = 64,
            **kwargs,
    ) -> List[List[float]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a dict of pairs of int(s) and list of float(s) with the evaluation outcome of batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = RegionPerturbation(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            batch_size=batch_size,
            **kwargs,
        )

    def process_batch(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: np.ndarray,
            a_batch: np.ndarray,
            s_batch: Optional[np.ndarray] = None,
            perturb_func: Callable = None,
            perturb_func_kwargs: Optional[Dict] = None,
    ):
        # TODO: use check function for this, but maybe already in __call__
        if perturb_func is None:
            raise ValueError("perturb_func must not be None")
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        # create array for scores of each perturbation step
        batch_size = x_batch.shape[0]
        n_steps = self.regions_evaluation
        preds = [[None for _ in range(n_steps)] for _ in range(batch_size)]

        # create generator for perturbed batches
        perturbation_generator = self.generate_perturbed_batches(
            x_batch=x_batch,
            a_batch=a_batch,
            n_steps=n_steps,
            patch_size=self.patch_size,
            perturb_func=perturb_func, 
           perturb_func_kwargs=perturb_func_kwargs,
        )

        # Predict on original unperturbed input x.
        # TODO: do this only if difference-flag is True
        x_input = model.shape_input(
            x=x_batch,
            shape=x_batch.shape,
            channel_first=True,
            batched=True,
        )
        y_batch_pred = model.predict(x=x_input, softmax_act=True)

        for step, x_batch_perturbed in enumerate(perturbation_generator):

            # Predict on perturbed input x.
            x_input = model.shape_input(
                x=x_batch_perturbed,
                shape=x_batch.shape,
                channel_first=True,
                batched=True,
            )

            y_batch_pred_perturb = model.predict(x=x_input, softmax_act=True)

            # TODO: get rid of for loop
            iterator = zip(y_batch, y_batch_pred_perturb)
            for ix, (y, y_pred_perturb) in enumerate(iterator):
                y_pred_perturb = float(y_pred_perturb[y])
                y_pred = float(y_batch_pred[ix, y])
                # TODO: flag difference output
                preds[ix][step] = y_pred - y_pred_perturb

        return preds

    def generate_perturbed_batches(
            self,
            x_batch: np.ndarray,
            a_batch: np.ndarray,
            n_steps: int,
            patch_size: int, # TODO: add support for type Sequence[int]
            perturb_func: Callable,
            perturb_func_kwargs: Dict[str, Any],
    ) -> Iterator[np.ndarray]:

        patches_batch, pad_width = self.get_ordered_patches(
            x_batch=x_batch,
            a_batch=a_batch,
            patch_size=patch_size,
            max_patches=n_steps,
            overlapping=False,
            return_indices=False,
        )

        # TODO: assert max_patches == n_steps

        # Increasingly perturb the input and store the decrease in function value.
        x_batch_perturbed = x_batch.copy()
        x_batch_perturbed_last = x_batch_perturbed.copy()
        for step in range(n_steps):

            # Select patches of current step for each instance in batch.
            # TODO: There will be an IndexError raised if there's an instance for which
            # there are less patches available than desired steps. Decide what to do then.
            patches_batch_step = [
                patches_instance[step] for patches_instance in patches_batch
            ]
            # Pad x_perturbed. The mode should probably depend on the used perturb_func?
            x_perturbed_pad = utils.pad_array(
                x_batch_perturbed, pad_width, mode="edge",
                omit_first_axis=True, batched=True,
            )

            # Perturb complete batch inplace.
            perturb_funcs.perturb_batch_on_patches(
                arr=x_perturbed_pad,
                patches=patches_batch_step,
                perturb_func=perturb_func,
                **perturb_func_kwargs,
            )
            
            # Remove Padding
            x_batch_perturbed = utils.unpad_array(
                x_perturbed_pad, pad_width, omit_first_axis=True, batched=True,
            )
            
            asserts.assert_perturbation_caused_change(
                x=x_batch_perturbed_last, x_perturbed=x_batch_perturbed)
            x_batch_perturbed_last = x_batch_perturbed.copy()

            yield x_batch_perturbed

    def get_ordered_patches(
            self,
            x_batch: np.ndarray,
            a_batch: np.ndarray,
            patch_size: int,
            max_patches: int,
            overlapping: bool = False,
            return_indices: bool = False,  # This doesn't properly work yet.
                                           # TODO: either fix or remove this!
    ) -> Tuple[List[List[Patch]], int]:

        batch_size = x_batch.shape[0]

        # Pad input and attributions. This is needed to allow for any patch_size.
        pad_width = patch_size - 1
        x_batch_pad = utils.pad_array(
            x_batch, pad_width, mode="constant", omit_first_axis=True, batched=True,
        )
        a_batch_pad = utils.pad_array(
            a_batch, pad_width, mode="constant", omit_first_axis=True, batched=True,
        )

        # Create patches across whole input shape. Patches don't have batch axis.
        patches = []
        axis_iterators = [
            range(pad_width, x_batch_pad.shape[axis] - pad_width)
            # omit batch and channel dimensions
            for axis in range(2, x_batch_pad.ndim)
        ]

        # Add channel dimension to axis iterators
        # TODO: add boolean switch for this
        #axis_iterators.insert(0, range(x_batch_pad.shape[1]))
            
        for top_left_coords in itertools.product(*axis_iterators):

            # Create slice for patch.
            patch = utils.create_patch(
                patch_size=patch_size,
                coords=top_left_coords,
                expand_first_dim=False, # don't expand channel dimension
            )

            for channel_id in range(x_batch_pad.shape[1]):
                channel_patch = (channel_id, *patch)
                patches.append(channel_patch)

        # Sum attributions for each patch (on instance axis)
        a_sum_patches = np.zeros((batch_size, len(patches)))
        for patch_id, patch in enumerate(patches):
            # Add batch dimension to patch.
            patch_batch = (slice(None), *patch)
            a_patch_batch = a_batch_pad[patch_batch]
            # Take sum on all axes after batch axis.
            a_sum_patch_batch = np.sum(
                a_patch_batch, axis=tuple(range(1, a_patch_batch.ndim)),
            )
            try:
                a_sum_patches[:, patch_id] = a_sum_patch_batch.squeeze(axis=-1)
            except ValueError as e:
                a_sum_patches[:, patch_id] = a_sum_patch_batch

        # Order attributions by selected method
        if self.order == "random":
            # Create order for all instances
            patch_order_idx_batch = np.zeros((batch_size, len(patches)), dtype=int)
            for instance_id in range(batch_size):
                # Order attributions randomly for each instance.
                patch_order_idx_instance = np.arange(len(patches))
                np.random.shuffle(patch_order_idx_instance)
                patch_order_idx_batch[instance_id, :] = patch_order_idx_instance

        elif self.order == "morf":
            # Order attributions according to the most relevant first.
            patch_order_idx_batch = np.argsort(a_sum_patches)[:, ::-1]

        elif self.order == "lerf":
            # Order attributions according to the least relevant first.
            patch_order_idx_batch = np.argsort(a_sum_patches)

        else:
            raise ValueError(f"Invalid order passed: {self.order}")

        # Create ordered list of patches for each instance in batch.
        patches_batch = [
            [
                patches[patch_order_idx_instance]
                for patch_order_idx_instance in patch_order_idx_batch[instance_id]
            ]
            for instance_id in range(batch_size)
        ]

        # Remove overlapping patches if requested.
        if not overlapping:
            patches_batch = self.remove_overlapping_patches(
                patches_batch=patches_batch,
                x_shape=x_batch.shape,
                max_patches=max_patches,
            )

        # Limit returned patches to max_patches
        patches_batch = [
            patches_instance[:max_patches] for patches_instance in patches_batch
        ]

        # Return patches as arrays of indices if requested.
        if return_indices:
            patches_batch = utils.transform_patches_to_indices(
                patches=patches_batch, arr_shape=x_batch.shape,
            )

        return patches_batch, pad_width

    def remove_overlapping_patches(
            self,
            patches_batch: List[List[Patch]],
            x_shape: Tuple[int, ...],
            max_patches: int,
    ) -> List[List[Patch]]:

        blocked_mask = np.zeros(x_shape, dtype=bool)
        batch_size = len(patches_batch)
        patches_batch_no_overlap = [[] for _ in range(batch_size)]

        for instance_id, patches_instance in enumerate(patches_batch):
            for patch in patches_instance:
                # Create patch mask for current patch
                patch_mask = np.zeros(x_shape[1:], dtype=bool)
                patch_mask[patch] = True

                # Check for intersections and add patch if there are none
                intersected = blocked_mask[instance_id] & patch_mask
                if not intersected.any():
                    patches_batch_no_overlap[instance_id].append(patch)
                    blocked_mask[instance_id] = blocked_mask[instance_id] | patch_mask

                if len(patches_batch_no_overlap[instance_id]) >= max_patches:
                    break

            # fill up patch slices with empty patch slices
            # if there are not enough non-overlapping patches
            # TODO: check if this is acceptable
            #while len(patches_batch_no_overlap[instance_id]) < max_patches:
            #    patches_batch_no_overlap[instance_id].append([])

        return patches_batch_no_overlap
        
    def get_patches_batch_for_step(
            self,
            patches_batch: List[List[Patch]],
            step: int,
    ) -> List[Patch]:
        
        # For this step there's no patch available for at least one instance.
        # TODO: Discuss what to do here. Raise a warning?
        # Currently it is filled with an empty array to select no values.
        # That's why the checking of perturbation_caused_change is disabled.
        
        #except Exception as e:
        #    breakpoint()
            
        #finally:
        return patches_batch_step


class Selectivity(Metric):
    """
    Implementation of Selectivity test by Montavan et al., 2018.

    At each iteration, a patch of size 4 x 4 corresponding to the region with
    highest relevance is set to black. The plot keeps track of the function value
    as the features are being progressively removed and computes an average over
    a large number of examples.

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            patch_size (integer): The patch size for masking, default=8.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plotting.plot_selectivity_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_patch
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.patch_size = self.kwargs.get("patch_size", 8)
        self.last_results = {}
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the patch size for masking"
                    " 'patch_size'"
                ),
                citation=(
                    "Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. 'Methods for "
                    "interpreting and understanding deep neural networks.' Digital Signal "
                    "Processing 73 (2018): 1-15"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> Dict[int, List[float]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a dict of pairs of int(s) and list of float(s) with the evaluation outcome of batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Selectivity(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = {k: None for k in range(len(x_batch_s))}

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)), total=len(x_batch_s)
            )

        for sample, (x, y, a) in iterator:
            # Predict on input.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(
                model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
            )

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            patches = []
            sub_results = []
            x_perturbed = x.copy()

            # Pad input and attributions. This is needed to allow for any patch_size.
            pad_width = self.patch_size - 1
            x_pad = utils.pad_array(
                x, pad_width, mode="constant", omit_first_axis=True
            )
            a_pad = utils.pad_array(
                a, pad_width, mode="constant", omit_first_axis=True
            )

            # Get patch indices of sorted attributions (descending).
            # TODO: currently, image is split into a grid, with the patches as the grid elements.
            #  I.e., not all possible patches are considered. Consequently,
            #  not the patch with the highest relevance is chosen first, but the (static) grid element instead.
            #  This is changed now in regionperturbation. Would this change also be intended here?
            #  Leaving it as-is for now.
            #  IF this should be changed, overlapping patches need to be excluded, see RegionPerturbation
            att_sums = []
            axis_iterators = [
                range(pad_width, x_pad.shape[axis] - pad_width, self.patch_size)
                for axis in range(1, x_pad.ndim)
            ]
            for top_left_coords in itertools.product(*axis_iterators):
                # Create slice for patch.
                patch_slice = utils.create_patch(
                    patch_size=self.patch_size,
                    coords=top_left_coords,
                    expand_first_dim=True,
                )

                # Sum attributions for patch.
                att_sums.append(a_pad[patch_slice].sum())
                patches.append(patch_slice)

            # Order attributions according to the most relevant first.
            ordered_patches = [patches[p] for p in np.argsort(att_sums)[::-1]]

            # Increasingly perturb the input and store the decrease in function value.
            for patch_slice in ordered_patches:
                # Pad x_perturbed. The mode should probably depend on the used perturb_func?
                x_perturbed_pad = utils.pad_array(
                    x_perturbed, pad_width, mode="edge", omit_first_axis=True
                )

                # Perturb.
                x_perturbed_pad = self.perturb_func(
                    arr=x_perturbed_pad, patch=patch_slice, **self.kwargs,
                )

                # Remove Padding
                x_perturbed = utils.unpad_array(
                    x_perturbed_pad, pad_width, omit_first_axis=True
                )

                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(
                    model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
                )

                sub_results.append(y_pred_perturb)

            self.last_results[sample] = sub_results

        self.all_results.append(self.last_results)

        return self.last_results


class SensitivityN(Metric):
    """
    Implementation of Sensitivity-N test by Ancona et al., 2019.

    An attribution method satisfies Sensitivity-n when the sum of the attributions for any subset of features of
    cardinality n is equal to the variation of the output Sc caused removing the features in the subset. The test
    computes the correlation between sum of attributions and delta output.

    Pearson correlation coefficient (PCC) is computed between the sum of the attributions and the variation in the
    target output varying n from one to about 80% of the total number of features, where an average across a thousand
    of samples is reported. Sampling is performed using a uniform probability distribution over the features.

    References:
        1) Ancona, Marco, et al. "Towards better understanding of gradient-based attribution
        methods for deep neural networks." arXiv preprint arXiv:1711.06104 (2017).

    Current assumptions:
         - In the paper, they showcase a MNIST experiment where
         4x4 patches with black baseline value. Since we are taking ImageNet as dataset,
         we take 224/28=8 i.e., 8 times bigger patches to replicate the same analysis
         - Also, instead of replacing with a black pixel we take the mean of the
         neighborhood, so not to distort the image distribution completely.
         - I don't get why they have so high correlation in the paper, maybe using a better baseline_value?
         - Also I don't get why correlation is only reported positive?

    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_pearson.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform".
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            n_max_percentage (float): The percentage of features to iteratively evaluatede, fault=0.8.
            features_in_step (integer): The size of the step, default=1.
            max_steps_per_input (integer): The number of steps per input dimension, default=None.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plotting.plot_sensitivity_n_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.n_max_percentage = self.kwargs.get("n_max_percentage", 0.8)
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', the patch size for masking "
                    "'patch_size', similarity function 'similarity_func' and the number "
                    "of features to iteratively evaluate 'n_max_percentage'"
                ),
                citation=(
                    "Ancona, Marco, et al. 'Towards better understanding of gradient-based "
                    "attribution methods for deep neural networks.' arXiv preprint "
                    "arXiv:1711.06104 (2017)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = SensitivityN(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )
        if self.max_steps_per_input is not None:
            asserts.assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=x_batch_s.shape[2:],
            )
            self.set_features_in_step = utils.get_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                input_shape=x_batch_s.shape[2:],
            )

        max_features = int(0.8 * np.prod(x_batch_s.shape[2:]) // self.features_in_step)

        sub_results_pred_deltas = {k: [] for k in range(len(x_batch_s))}
        sub_results_att_sums = {k: [] for k in range(len(x_batch_s))}

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)), total=len(x_batch_s)
            )

        for sample, (x, y, a) in iterator:

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Get indices of sorted attributions (descending).
            a_indices = np.argsort(-a)

            # Predict on x.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(
                model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
            )

            att_sums = []
            pred_deltas = []
            x_perturbed = x.copy().flatten()

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    arr=x_perturbed,
                    indices=a_ix,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Sum attributions.
                att_sums.append(float(a[a_ix].sum()))

                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(
                    model.predict(x_input, softmax_act=True, **self.kwargs)[:, y]
                )
                pred_deltas.append(y_pred - y_pred_perturb)

            sub_results_att_sums[sample] = att_sums
            sub_results_pred_deltas[sample] = pred_deltas

        # Re-arrange sublists so that they are sorted by n.
        sub_results_pred_deltas_l = {k: [] for k in range(max_features)}
        sub_results_att_sums_l = {k: [] for k in range(max_features)}

        for k in range(max_features):
            for sublist1 in list(sub_results_pred_deltas.values()):
                sub_results_pred_deltas_l[k].append(sublist1[k])
            for sublist2 in list(sub_results_att_sums.values()):
                sub_results_att_sums_l[k].append(sublist2[k])

        # Measure similarity for each n.
        self.last_results = [
            self.similarity_func(
                a=sub_results_att_sums_l[k], b=sub_results_pred_deltas_l[k]
            )
            for k in range(max_features)
        ]
        self.all_results.append(self.last_results)

        return self.last_results
