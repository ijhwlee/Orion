"""Orion Core module.

This module defines the Orion Class, which is responsible for the
main anomaly detection functionality, as well as the interaction
with the underlying MLBlocks pipelines.
"""
import json
import logging
import os
import pickle
import sys
from typing import List, Union

import pandas as pd
from mlblocks import MLPipeline
from mlblocks import discovery

from orion.evaluation import CONTEXTUAL_METRICS as METRICS

LOGGER = logging.getLogger(__name__)

_ORION_PIPELINES_PATHS = [
    os.path.join(os.getcwd(), 'mlpipelines'),
    os.path.join(sys.prefix, 'orion'),
    os.path.join(sys.prefix, 'orion', 'pipelines'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'orion', 'pipelines', 'verified', 'aer'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'orion', 'pipelines', 'verified', 'arima'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'orion', 'pipelines', 'verified', 'azure'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'orion', 'pipelines', 'verified', 'dense_autoencoder'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'orion', 'pipelines', 'verified', 'lstm_autoencoder'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'orion', 'pipelines', 'verified', 'lstm_dynamic_threshold'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'orion', 'pipelines', 'verified', 'tadgan'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'orion', 'pipelines', 'verified', 'vae'),
    os.path.join(os.getcwd(), 'mlblocks_pipelines'),    # legacy
    os.path.join(sys.prefix, 'mlblocks_pipelines'),    # legacy
]
_ORION_PRIMITIVES_PATHS = [
    os.path.join(os.getcwd(), 'primitives'),
    os.path.join(sys.prefix, 'orion'),
    os.path.join(sys.prefix, 'orion', 'primitives'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'orion', 'primitives', 'jsons'),
    os.path.join(os.getcwd(), 'mlblocks_primitives'),    # legacy
    os.path.join(sys.prefix, 'mlblocks_primitives'),    # legacy
]

class Orion:
    """Orion Class.

    The Orion Class provides the main anomaly detection functionalities
    of Orion and is responsible for the interaction with the underlying
    MLBlocks pipelines.

    Args:
        pipeline (str, dict or MLPipeline):
            Pipeline to use. It can be passed as:
                * An ``str`` with a path to a JSON file.
                * An ``str`` with the name of a registered pipeline.
                * An ``MLPipeline`` instance.
                * A ``dict`` with an ``MLPipeline`` specification.
        hyperparameters (dict):
            Additional hyperparameters to set to the Pipeline.
    """

    PIPELINES_DIR = tuple(
        dirname
        for dirname, _, _ in os.walk(os.path.join(os.path.dirname(__file__), 'pipelines'))
        if os.path.exists(os.path.join(dirname, os.path.basename(dirname) + '.json'))
    )
    PIPELINES = tuple(os.path.basename(pipeline) for pipeline in PIPELINES_DIR)

    DEFAULT_PIPELINE = 'lstm_dynamic_threshold'

    def add_orion_primitives_paths(self):
        for path in _ORION_PRIMITIVES_PATHS:
            #print("[DEBUG-hwlee]orion.core.add_orion_primitives_paths: path = {0}".format(path))
            if os.path.isdir(path):
                discovery.add_primitives_path(path)
                #print("[DEBUG-hwlee]orion.core.add_orion_primitives_paths: path({0}) is ADDED".format(path))

    def _get_mlpipeline(self):
        pipeline = None
        pipeline_name = self._pipeline
        if isinstance(pipeline_name, str) and os.path.isfile(pipeline_name):
            with open(pipeline_name) as json_file:
                pipeline = json.load(json_file)

        if pipeline is None:
            for base_path in _ORION_PIPELINES_PATHS:
                # first check if there is json file with given name
                filename = pipeline_name + '.json'
                json_path = os.path.join(base_path, filename)

                #print("[DEBUG-hwlee]Orion.core._get_mlpipeline: json_path = {0}".format(json_path))
                #print("[DEBUG-hwlee]Orion.core._get_mlpipeline: os.path.isfile(json_path) = {0}".format(os.path.isfile(json_path)))
                if os.path.isfile(json_path):
                    with open(json_path) as json_file:
                        pipeline = json.load(json_file)
                    break

        if pipeline is None:
            print("[DEBUG-hwlee]Critical Error not possible to load pipeline at {0}: {1}".format(__file__, 76))
        self.add_orion_primitives_paths()
        mlpipeline = MLPipeline(pipeline)
        if self._hyperparameters:
            mlpipeline.set_hyperparameters(self._hyperparameters)

        return mlpipeline

    def __init__(self, pipeline: Union[str, dict, MLPipeline] = None,
                 hyperparameters: dict = None):
        self._pipeline = pipeline or self.DEFAULT_PIPELINE
        self._hyperparameters = hyperparameters
        self._mlpipeline = self._get_mlpipeline()
        self._fitted = False

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__) and
            self._pipeline == other._pipeline and
            self._hyperparameters == other._hyperparameters and
            self._fitted == other._fitted
        )

    def __repr__(self):
        if isinstance(self._pipeline, MLPipeline):
            pipeline = '\n'.join(
                '    {}'.format(primitive) for primitive in self._pipeline.to_dict()['primitives'])

        elif isinstance(self._pipeline, dict):
            pipeline = '\n'.join(
                '    {}'.format(primitive) for primitive in self._pipeline['primitives'])

        else:
            pipeline = '    {}'.format(self._pipeline)

        hyperparameters = None
        if self._hyperparameters is not None:
            hyperparameters = '\n'.join(
                '    {}: {}'.format(step, value) for step, value in self._hyperparameters.items())

        return (
            'Orion:\n{}\n'
            'hyperparameters:\n{}\n'
        ).format(
            pipeline,
            hyperparameters
        )

    def fit(self, data: pd.DataFrame, **kwargs):
        """Fit the pipeline to the given data.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
        """
        if not self._fitted:
            self._mlpipeline = self._get_mlpipeline()

        self._mlpipeline.fit(data, **kwargs)
        self._fitted = True

    def _get_outputs_spec(self):
        outputs_spec = ["default"]
        try:
            visualization_outputs = self._mlpipeline.get_output_names('visualization')
            outputs_spec.append('visualization')
        except ValueError:
            visualization_outputs = []

        return outputs_spec, visualization_outputs

    @staticmethod
    def _build_events_df(events):
        events = pd.DataFrame(list(events), columns=['start', 'end', 'severity'])
        events['start'] = events['start'].astype('int64')
        events['end'] = events['end'].astype('int64')

        return events

    def _detect(self, method, data, visualization=False, **kwargs):
        if visualization:
            outputs_spec, visualization_names = self._get_outputs_spec()
        else:
            outputs_spec = 'default'

        outputs = method(data, output_=outputs_spec, **kwargs)

        if visualization:
            if visualization_names:
                events = outputs[0]
                visualization_outputs = outputs[-len(visualization_names):]
                visualization_dict = dict(zip(visualization_names, visualization_outputs))
            else:
                events = outputs
                visualization_dict = {}

            return self._build_events_df(events), visualization_dict

        return self._build_events_df(outputs)

    def detect(self, data: pd.DataFrame, visualization: bool = False) -> pd.DataFrame:
        """Detect anomalies in the given data..

        If ``visualization=True``, also return the visualization
        outputs from the MLPipeline object.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            visualization (bool):
                If ``True``, also capture the ``visualization`` named
                output from the ``MLPipeline`` and return it as a second
                output.

        Returns:
            DataFrame or tuple:
                If visualization is ``False``, it returns the events
                DataFrame. If visualization is ``True``, it returns a
                tuple containing the events DataFrame followed by the
                visualization outputs dict.
        """
        return self._detect(self._mlpipeline.predict, data, visualization)

    def fit_detect(self, data: pd.DataFrame, visualization: bool = False,
                   **kwargs) -> pd.DataFrame:
        """Fit the pipeline to the data and then detect anomalies.

        This method is functionally equivalent to calling ``fit(data)``
        and later on ``detect(data)`` but with the difference that
        here the ``MLPipeline`` is called only once, using its ``fit``
        method, and the output is directly captured without having
        to execute the whole pipeline again during the ``predict`` phase.

        If ``visualization=True``, also return the visualization
        outputs from the MLPipeline object.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            visualization (bool):
                If ``True``, also capture the ``visualization`` named
                output from the ``MLPipeline`` and return it as a second
                output.

        Returns:
            DataFrame or tuple:
                If visualization is ``False``, it returns the events
                DataFrame. If visualization is ``True``, it returns a
                tuple containing the events DataFrame followed by the
                visualization outputs dict.
        """
        if not self._fitted:
            self._mlpipeline = self._get_mlpipeline()

        result = self._detect(self._mlpipeline.fit, data, visualization, **kwargs)
        self._fitted = True

        return result

    def save(self, path: str):
        """Save this object using pickle.

        Args:
            path (str):
                Path to the file where the serialization of
                this object will be stored.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path: str):
        """Load an Orion instance from a pickle file.

        Args:
            path (str):
                Path to the file where the instance has been
                previously serialized.

        Returns:
            Orion

        Raises:
            ValueError:
                If the serialized object is not an Orion instance.
        """
        with open(path, 'rb') as pickle_file:
            orion = pickle.load(pickle_file)
            if not isinstance(orion, cls):
                raise ValueError('Serialized object is not an Orion instance')

            return orion

    def evaluate(self, data: pd.DataFrame, ground_truth: pd.DataFrame, fit: bool = False,
                 train_data: pd.DataFrame = None, metrics: List[str] = METRICS) -> pd.Series:
        """Evaluate the performance against ground truth anomalies.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            ground_truth (DataFrame):
                Ground truth anomalies passed as a ``pandas.DataFrame``
                containing two columns: start and stop.
            fit (bool):
                Whether to fit the pipeline before evaluating it.
                Defaults to ``False``.
            train_data (DataFrame):
                Training data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
                If not given, the pipeline is fitted on ``data``.
            metrics (list):
                List of metrics to used passed as a list of strings.
                If not given, it defaults to all the Orion metrics.

        Returns:
            Series:
                ``pandas.Series`` containing one element for each
                metric applied, with the metric name as index.
        """
        if not fit:
            method = self._mlpipeline.predict
        else:
            if not self._fitted:
                mlpipeline = self._get_mlpipeline()
            else:
                mlpipeline = self._mlpipeline

            if train_data is not None:
                # Fit first and then predict
                mlpipeline.fit(train_data)
                method = mlpipeline.predict
            else:
                # Fit and predict at once
                method = mlpipeline.fit

        events = self._detect(method, data)

        scores = {
            metric: METRICS[metric](ground_truth, events, data=data)
            for metric in metrics
        }

        return pd.Series(scores)
