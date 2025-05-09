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

SINCNET_DEFAULTS = {"ksize": 251, "stride": 10}
SINCNET_DEFAULTS = {"ksize": 321, "stride": 56}

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

    # SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: Optional[dict] = None,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        frame_sec:float = 0.
    ):
        super(PyanNet_nn, self).__init__()

        self.specifications = None
        # sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        # sincnet["sample_rate"] = sample_rate
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

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
        print('diar.self.hparams', self.hparams)
        self._sincnet = SincNetPool(sample_rate=16000, **SINCNET_DEFAULTS)


        self.post_pool = None
        if frame_sec > 0:
            # 기존 스텝 크기 계산 (대략 0.017307692초)
            base_step = 0.017307692

            # 목표 스텝 크기(예: 0.102초)에 맞는 풀링 커널 크기 계산
            pool_kernel: int = math.ceil(frame_sec / base_step)
            pool_stride: int = pool_kernel

            self.post_pool = (
                nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, ceil_mode=True)
            )
            print(f"PyanNet: {pool_kernel=}, {pool_stride=}, target_step={frame_sec}, base_step={base_step}")

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
        n_frames = self._sincnet.num_frames(num_samples)

        # post_pool이 있으면 풀링에 의한 프레임 수 감소를 계산
        if self.post_pool is not None:
            pool_kernel = self.post_pool.kernel_size[0] if isinstance(self.post_pool.kernel_size, tuple) else self.post_pool.kernel_size
            pool_stride = self.post_pool.stride[0] if isinstance(self.post_pool.stride, tuple) else self.post_pool.stride
            pool_padding = self.post_pool.padding if hasattr(self.post_pool, 'padding') else 0

            # 평균 풀링 후 프레임 수 계산 공식
            # out_length = (in_length + 2*padding - kernel_size) / stride + 1
            n_frames = (n_frames + 2*pool_padding - pool_kernel) // pool_stride + 1

        return n_frames


    @cached_property
    def receptive_field(self) -> SlidingWindow:
        """(Internal) frames"""
        _receptive_field_size = self._sincnet.receptive_field_size(num_frames=1)
        _receptive_field_step = (
            self._sincnet.receptive_field_size(num_frames=2) - _receptive_field_size
        )
        receptive_field_start = (
            self._sincnet.receptive_field_center(frame=0) - (_receptive_field_size - 1) / 2
        )
        sr = 16000

        # 기본값
        start_sec = receptive_field_start / sr
        duration_sec = _receptive_field_size / sr
        step_sec = _receptive_field_step / sr

        # 만약 post_pool이 존재한다면, 스텝 크기를 조정
        if self.post_pool is not None:
            pool_kernel = self.post_pool.kernel_size[0]
            step_sec = step_sec * pool_kernel  # 풀링 커널 크기에 비례하여 스텝 증가
            # duration은 변경하지 않음 - 원래 receptive field 크기 유지

        return SlidingWindow(
            start=start_sec,
            duration=duration_sec,
            step=step_sec,  # 0.102초(약 0.0173 * 6)로 설정될 것임
        )



    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        return self._sincnet.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """
        return self._sincnet.receptive_field_center(frame=frame)

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

        if self.post_pool is not None:
            outputs = self.post_pool(outputs)

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

    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: Optional[dict] = None,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("sincnet", "lstm", "linear")

        # print( 'self.hparams.sincnet', self.hparams.sincnet)
        # print( 'pyan.hparams.sincnet', self.hparams)

        self.sincnet = SincNetPool(sample_rate=16000, **SINCNET_DEFAULTS)
        self.post_pool = None

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

        value = self.sincnet.num_frames(num_samples)
        print(f"PyanNet: {value=}, {num_samples=}")
        return value

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        value = self.sincnet.receptive_field_size(num_frames=num_frames)
        # print(f"PyanNet.receptive_field_size: {value=}, {num_frames=}")
        return value

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        value = self.sincnet.receptive_field_center(frame=frame)
        # print(f"PyanNet.receptive_field_center: {value=}, {frame=}")
        return value

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.sincnet(waveforms)
        if self.post_pool is not None:
            outputs = self.post_pool(outputs)

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
