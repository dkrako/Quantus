"""This module implements the base class for creating evaluation measures."""
import math
import warnings
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..helpers import utils
from ..helpers import asserts
from ..helpers.model_interface import ModelInterface
from ..helpers import warn_func


class Metric:
    """
    Implementation base Metric class.
    """

    @asserts.attributes_check
    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func: Optional[Callable],
        normalise_func_kwargs: Optional[Dict[str, Any]],
        softmax: bool,
        default_plot_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
        **kwargs,
    ):
        """
        Initialise the Metric base class.

        Each of the defined metrics in Quantus, inherits from Metric base class.

        A child metric can benefit from the following class methods:
        - __call__(): Will call general_preprocess(), apply evaluate_instance() on each
                      instance and finally call custom_preprocess().
                      To use this method the child Metric needs to implement
                      evaluate_instance().
        - general_preprocess(): Prepares all necessary data structures for evaluation.
                                Will call custom_preprocess() at the end.

        Parameters
        ----------
        abs (boolean): Indicates whether absolute operation is applied on the attribution.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed.

        """
        # Run deprecation warnings.
        warn_func.deprecation_warnings(kwargs)
        asserts.check_kwargs(kwargs)

        self.abs = abs
        self.normalise = normalise
        self.softmax = softmax
        self.normalise_func = normalise_func

        if normalise_func_kwargs is None:
            normalise_func_kwargs = {}
        self.normalise_func_kwargs = normalise_func_kwargs

        self.default_plot_func = default_plot_func
        self.disable_warnings = disable_warnings
        self.display_progressbar = display_progressbar

        self.last_results = []
        self.all_results = []

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        target: Union[str, np.ndarray],
        channel_first: Optional[bool],
        explain_func: Optional[Callable],
        explain_func_kwargs: Optional[Dict[str, Any]],
        model_predict_kwargs: Optional[Dict],
        softmax: Optional[bool],
        device: Optional[str] = None,
        **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        evaluate_instance() on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        Parameters
        ----------
        model: a torch model e.g., torchvision.models that is subject to explanation
        x_batch: a np.ndarray which contains the input data that are explained
        y_batch: a np.ndarray which contains the output labels that are explained
        a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
        s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
        channel_first (boolean, optional): Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func (callable): Callable generating attributions.
        explain_func_kwargs (dict, optional): Keyword arguments to be passed to explain_func on call.
        device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        model_predict_kwargs (dict, optional): Keyword arguments to be passed to the model's predict method.

        Returns
        -------
        last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
        --------
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
        >> metric = Metric(abs=True, normalise=False)
        >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """
        # Run deprecation warnings.
        # TODO: Move this type of warning into the for loop.
        #       We want to be flexible and add custom metric call kwargs to
        #       the calling of self.evaluate_instance().
        #       Check there if the implemented method of the child class
        #       has this keyword specified. If not, raise a warning.
        warn_func.deprecation_warnings(kwargs)
        asserts.check_kwargs(kwargs)

        model, x_batch, y_batch, a_batch, s_batch = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            target=target,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
        )

        # Create progress bar if desired.
        iterator = tqdm(
            enumerate(zip(x_batch, y_batch, a_batch, s_batch)),
            total=len(x_batch),
            disable=not self.display_progressbar,
        )
        self.last_results = [None for _ in x_batch]
        for instance_id, (x_instance, y_instance, a_instance, s_instance) in iterator:
            result = self.evaluate_instance(
                model=model,
                x=x_instance,
                y=y_instance,
                a=a_instance,
                s=s_instance,
                **kwargs,
            )
            self.last_results[instance_id] = result

        # Call post-processing
        self.custom_postprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
        )

        self.all_results.append(self.last_results)
        return self.last_results

    @abstractmethod
    def evaluate_instance(
        self,
        model: ModelInterface,
        x_instance: np.ndarray,
        y_instance: Optional[np.ndarray] = None,
        a_instance: Optional[np.ndarray] = None,
        s_instance: Optional[np.ndarray] = None,
    ) -> Any:
        """
        This method needs to be implemented to use __call__().

        Gets model and data for a single instance as input, returns result.
        """
        raise NotImplementedError()

    def general_preprocess(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        target: Union[str, np.ndarray],
        channel_first: Optional[bool],
        explain_func: Optional[Callable],
        explain_func_kwargs: Optional[Dict[str, Any]],
        model_predict_kwargs: Optional[Dict],
        softmax: bool,
        device: Optional[str],
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, ModelInterface, Dict[str, Any]
    ]:
        """
        Prepares all necessary variables for evaluation.

        - Reshapes data to channel first layout.
        - Wraps model into ModelInterface.
        - Creates attribtions if necessary.
        - Expands attributions to data shape (adds channel dimension).
        - Calls custom_preprocess().
        - Normalises attributions if desired.
        - Takes absolute of attributions if desired.
        - If no segmentation s_batch given, creates list of Nones with as many
          elements as there are data instances.
        """
        # Reshape input batch to channel first order:
        if not isinstance(channel_first, bool):  # None is not a boolean instance.
            channel_first = utils.infer_channel_first(x_batch)
        x_batch = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            # Use attribute value if not passed explicitly.
            if softmax is None:
                softmax = self.softmax
            model = utils.get_wrapped_model(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                predict_kwargs=model_predict_kwargs,
            )

        # Save as attribute, some metrics need it during processing.
        self.explain_func = explain_func
        if explain_func_kwargs is None:
            explain_func_kwargs = {}
        self.explain_func_kwargs = explain_func_kwargs

        if a_batch is None:

            if target != 'true':
                raise NotImplementedError()

            # Asserts.
            asserts.assert_explain_func(explain_func=self.explain_func)

            # Generate explanations.
            a_batch = self.explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.explain_func_kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch, a_batch=a_batch)

        # Call custom pre-processing from inheriting class.
        model, x_batch, y_batch, a_batch, s_batch = self.custom_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
        )

        # Normalise with specified keyword arguments if requested.
        if self.normalise:
            a_batch = self.normalise_func(
                a=a_batch,
                # TODO note: this assumes we always have a batch axis. that ok?
                normalized_axes=list(range(np.ndim(a_batch)))[1:],
                **self.normalise_func_kwargs,
            )

        # Take absolute if requested.
        if self.abs:
            a_batch = np.abs(a_batch)

        # This is needed for iterator (zipped over x_batch, y_batch, a_batch, s_batch)
        if s_batch is None:
            s_batch = [None for _ in x_batch]

        return model, x_batch, y_batch, a_batch, s_batch

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Implement this method if you need custom preprocessing of data,
        model alteration or simply for creating/initializing additional attributes.
        """
        return model, x_batch, y_batch, a_batch, s_batch

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Implement this method if you need custom postprocessing of results or
        additional attributes.
        """
        pass

    @property
    def interpret_scores(self) -> None:
        """

        Returns
        -------

        """
        print(self.__init__.__doc__.split(".")[1].split("References")[0])
        # print(self.__call__.__doc__.split("callable.")[1].split("Parameters")[0])

    @property
    def get_params(self) -> dict:
        """
        List parameters of metric.
        Returns: a dictionary with attributes if not excluded from pre-determined list
        -------

        """
        attr_exclude = [
            "args",
            "kwargs",
            "all_results",
            "last_results",
            "default_plot_func",
            "disable_warnings",
            "display_progressbar",
        ]
        return {k: v for k, v in self.__dict__.items() if k not in attr_exclude}

    def set_params(self, key: str, value: Any) -> dict:
        """
        Set a parameter of a metric.
        Parameters
        ----------
        key: attribute of metric to mutate
        value: value to update the key with

        -------
        Returns: the updated dictionary.

        """
        self.kwargs[key] = value
        return self.kwargs

    def plot(
        self,
        plot_func: Union[Callable, None] = None,
        show: bool = True,
        path_to_save: Union[str, None] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Basic plotting functionality for Metric class.
        The user provides a plot_func (Callable) that contains the actual plotting logic (but returns None).

        Parameters
        ----------
        plot_func: a Callable with the actual plotting logic.
        show: a boolean to state if the plot shall be shown.
        path_to_save: a string that specifies the path to save file.
        args: an optional with additional arguments.
        kwargs: an optional dict with additional arguments.

        Returns: None.
        -------

        """

        # Get plotting func if not provided.
        if plot_func is None:
            plot_func = self.default_plot_func

        # Asserts.
        asserts.assert_plot_func(plot_func=plot_func)

        # Plot!
        plot_func(*args, **kwargs)

        if show:
            plt.show()

        if path_to_save:
            plt.savefig(fname=path_to_save, dpi=400)

        return None


class BatchedMetric(Metric):
    """
    Implementation base BatchedMetric class.
    """

    @asserts.attributes_check
    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func: Optional[Callable],
        normalise_func_kwargs: Optional[Dict[str, Any]],
        softmax: bool,
        default_plot_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
        **kwargs,
    ):
        """
        Initialise the Metric base class.

        Each of the defined metrics in Quantus, inherits from Metric base class.

        A child metric can benefit from the following class methods:
        - __call__(): Will call general_preprocess(), apply evaluate_instance() on each
                      instance and finally call custom_preprocess().
                      To use this method the child Metric needs to implement
                      evaluate_instance().
        - general_preprocess(): Prepares all necessary data structures for evaluation.
                                Will call custom_preprocess() at the end.

        Parameters
        ----------
        abs (boolean): Indicates whether absolute operation is applied on the attribution.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed.

        """

        # Initialize super-class with passed parameters
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            softmax=softmax,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        target: Union[str, np.ndarray],
        channel_first: Optional[bool],
        explain_func: Optional[Callable],
        explain_func_kwargs: Optional[Dict[str, Any]],
        model_predict_kwargs: Optional[Dict],
        softmax: Optional[bool],
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        evaluate_instance() on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        Parameters
        ----------
        model: a torch model e.g., torchvision.models that is subject to explanation
        x_batch: a np.ndarray which contains the input data that are explained
        y_batch: a np.ndarray which contains the output labels that are explained
        a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
        s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
        channel_first (boolean, optional): Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func (callable): Callable generating attributions.
        explain_func_kwargs (dict, optional): Keyword arguments to be passed to explain_func on call.
        device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        model_predict_kwargs (dict, optional): Keyword arguments to be passed to the model's predict method.
        batch_size (int): batch size for evaluation, default = 64.

        Returns
        -------
        last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
        --------
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
        >> metric = Metric(abs=True, normalise=False)
        >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """
        # Run deprecation warnings.
        # TODO: Move this type of warning into the for loop.
        #       We want to be flexible and add custom metric call kwargs to
        #       the calling of self.evaluate_instance().
        #       Check there if the implemented method of the child class
        #       has this keyword specified. If not, raise a warning.
        #warn_func.deprecation_warnings(kwargs)
        #asserts.check_kwargs(kwargs)

        model, x, y, a, s = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            target=target,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
        )

        # create generator for generating batches
        batch_generator = self.generate_batches(
            x, y, a, s,
            batch_size=batch_size,
            display_progressbar=self.display_progressbar,
        )

        n_batches = self.get_number_of_batches(arr=x, batch_size=batch_size)
        #self.last_results = [None for _ in range(n_batches)]
        self.last_results = []
        # We use a tailing underscore to prevent confusion with the passed parameters.
        # TODO: rename kwargs of __call__() method accordingly or else this still will be confusing.
        for x_batch_, y_batch_, a_batch_, s_batch_ in batch_generator:
            result = self.process_batch(
                model=model,
                x_batch=x_batch_,
                y_batch=y_batch_,
                a_batch=a_batch_,
                s_batch=s_batch_,
                target=target,
                **kwargs,
            )
            self.last_results.extend(result)

        # Call post-processing
        self.custom_postprocess(
            model=model,
            x_batch=x,
            y_batch=y,
            a_batch=a,
            s_batch=s,
        )

        self.all_results.append(self.last_results)
        return self.last_results

    @abstractmethod
    def process_batch(
            self,
            x_batch: np.ndarray,
            y_batch: np.ndarray,
            a_batch: np.ndarray,
            s_batch: Optional[np.ndarray],
            target: Union[str, np.ndarray],
            **kwargs,
    ):
        raise NotImplementedError()

    @staticmethod
    def get_number_of_batches(arr: Sequence, batch_size: int):
        return math.ceil(len(arr)/batch_size)

    def generate_batches(
            self,
            *iterables: np.ndarray, batch_size: int,
            display_progressbar: bool = False,
    ):
        if iterables[0] is None:
            raise ValueError("first iterable must not be None!")

        iterables = list(iterables)
        n_instances = len(iterables[0])
        n_batches = self.get_number_of_batches(iterables[0], batch_size=batch_size)

        # check if any of the iterables is None and replace with list of None
        for i in range(len(iterables)):
            if iterables[i] is None:
                iterables[i] = [None for _ in range(n_instances)]

        if not all(len(iterable) == len(iterables[0]) for iterable in iterables):
            raise ValueError("number of instances needs to be equal for all iterables")

        iterator = tqdm(
            range(0, n_batches),
            total=n_batches,
            disable=not display_progressbar,
        )

        for batch_idx in iterator:
            batch_start = batch_size * batch_idx
            batch_end = min(batch_size * (batch_idx + 1), n_instances)
            batch = tuple(iterable[batch_start:batch_end]  for iterable in iterables)
            yield batch


class PerturbationMetric(Metric):
    """
    Implementation base PertubationMetric class.

    This metric has additional attributes for perturbations.
    """

    @asserts.attributes_check
    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func: Optional[Callable],
        normalise_func_kwargs: Optional[Dict[str, Any]],
        softmax: bool,
        perturb_func: Callable,
        perturb_func_kwargs: Optional[Dict[str, Any]],
        default_plot_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
        **kwargs,
    ):

        # Initialize super-class with passed parameters
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            softmax=softmax,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        self.perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        self.perturb_func_kwargs = perturb_func_kwargs


class BatchedPerturbationMetric(BatchedMetric):
    """
    Implementation base BatchedPertubationMetric class.

    This batched metric has additional attributes for perturbations.
    """

    @asserts.attributes_check
    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func: Optional[Callable],
        normalise_func_kwargs: Optional[Dict[str, Any]],
        softmax: bool,
        perturb_func: Callable,
        perturb_func_kwargs: Optional[Dict[str, Any]],
        default_plot_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
        **kwargs,
    ):

        # Initialize super-class with passed parameters
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            softmax=softmax,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        self.perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        self.perturb_func_kwargs = perturb_func_kwargs

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        channel_first: Optional[bool],
        explain_func: Optional[Callable],
        explain_func_kwargs: Optional[Dict[str, Any]],
        model_predict_kwargs: Optional[Dict],
        softmax: Optional[bool],
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> Union[int, float, list, dict, None]:
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            perturb_func=self.perturb_func,
            perturb_func_kwargs=self.perturb_func_kwargs,
            **kwargs,
        )
