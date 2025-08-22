from functools import lru_cache
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.audio.core.task import Problem
# from pyannote.audio.models.blocks.sincnet import SincNet
from diarizers.models.sincnet_pooling import SincNetPool
from pyannote.audio.utils.params import merge_dict
from pyannote.core.utils.generators import pairwise


from functools import cached_property
from pyannote.core import SlidingWindow

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        self.__dict__ = self



class PyanNet_nn(torch.nn.Module):
    """PyanNet segmentation model adapted from PyanNet model used in pyannote.
    Inherits from nn.Module (no more dependency to pytorch Lightning)
    Doesn't need the task attribute anymore.

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    def __init__(
        self,
        sincnet: dict,
        lstm: dict|None = None,
        linear: dict|None = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ):
        super().__init__()

        LSTM_DEFAULTS = {
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "monolithic": True,
            "dropout": 0.0,
        }
        LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}
        # sincnet = sincnet or { "ksize": 251, "stride": 10}

        self.specifications = None
        # sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        # sincnet["sample_rate"] = sample_rate
        lstm = merge_dict(LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(LINEAR_DEFAULTS, linear)

        self.hparams = Dict({
            "linear": {"hidden_size": 128, "num_layers": 2},
            "lstm": {
                "hidden_size": 128,
                "num_layers": 4,
                "bidirectional": True,
                "monolithic": True,
                "dropout": 0.0,
                "batch_first": True,
            },
            "num_channels": 1,
            "sample_rate": 16000,
            # "sincnet": {"stride": 10, "sample_rate": 16000, },
        })
        print(f"{sincnet=}")
        self.sincnet = sincnet
        self._sincnet_pool = SincNetPool(sample_rate=16000, stride=sincnet['stride'], ksize=sincnet['ksize'])

        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(60, **multi_layer_lstm)

        else:
            assert False, "not called"
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        60 if i == 0 else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm,
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            # print(f'pyannet: {self.specifications.num_powerset_classes=}')
            return self.specifications.num_powerset_classes
        else:
            print(f'pyannet: {self.specifications.classes=}')
            return len(self.specifications.classes)

    def default_activation(
        self,
    ):
        print('diar.specifications.problem', self.specifications.problem)
        if self.specifications.problem == Problem.BINARY_CLASSIFICATION:
            return nn.Sigmoid()

        elif self.specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
            return nn.LogSoftmax(dim=-1)

        elif self.specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            return nn.Sigmoid()
        else:
            msg = "TODO: implement default activation for other types of problems"
            raise NotImplementedError(msg)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()
        print(f'classfier: ({in_features}, {self.dimension})')

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """
        # SincNet에서 처리된 기본 프레임 수 계산
        n_frames = self._sincnet_pool.num_frames(num_samples)
        return n_frames


    @cached_property
    def receptive_field(self) -> SlidingWindow:
        """(Internal) frames"""
        start, size, step = self._sincnet_pool.receptive_field()
        sr = 16000

        return SlidingWindow(
            start = start / sr,
            duration = size / sr,
            step = step / sr,
        )

    # def receptive_field_size(self, num_frames: int = 1) -> int:
    #     return self._sincnet._receptive_field_size(num_frames=num_frames)
    #
    # def receptive_field_center(self, frame: int = 0) -> int:
    #     return self._sincnet._receptive_field_center(frame=frame)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample): (1, 1, 36000)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self._sincnet_pool(waveforms) # (1, 60, 130)

        if self.hparams.lstm["monolithic"]:
            outputs = rearrange(outputs, "batch feature frame -> batch frame feature") # (1, 130, 60)
            outputs, _ = self.lstm(outputs) # (1, 130, 256)
        else:
            assert False, "lstm must be monolithic"
            outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))
            # (1, 130, 128)

        return self.activation(self.classifier(outputs)) # (1, 130, num_classes)



#
# from pyannote
#
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
# from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict


class PyanNet(Model):
    """PyanNet segmentation model

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """



    def __init__(
        self,
        sincnet: dict,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}
        SINCNET_DEFAULTS = {"stride": 10}
        LSTM_DEFAULTS = {
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "monolithic": True,
            "dropout": 0.0,
        }
        lstm = merge_dict(LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("sincnet", "lstm", "linear")

        sincnet = sincnet or SINCNET_DEFAULTS
        # print( 'self.hparams.sincnet', self.hparams.sincnet)
        # print( 'pyan.hparams.sincnet', self.hparams)
        self.sincnet = sincnet
        self._sincnet = SincNetPool(sample_rate=16000, **sincnet)

        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(60, **multi_layer_lstm)

        else:
            assert False, "lstm must be monolithic"
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        60
                        if i == 0
                        else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """

        value = self._sincnet.num_frames(num_samples)
        print(f"PyanNet: {value=}, {num_samples=}")
        return value

    # def receptive_field_size(self, num_frames: int = 1) -> int:
    #     return self.sincnet._receptive_field_size(num_frames=num_frames)
    # def receptive_field_center(self, frame: int = 0) -> int:
    #     return self.sincnet._receptive_field_center(frame=frame)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self._sincnet(waveforms)

        if self.hparams.lstm["monolithic"]:
            outputs, _ = self.lstm(
                rearrange(outputs, "batch feature frame -> batch frame feature")
            )
        else:
            assert False, "lstm must be monolithic"
            outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
