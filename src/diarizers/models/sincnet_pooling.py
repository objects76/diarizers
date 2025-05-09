
from functools import lru_cache

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import asteroid_filterbanks

from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames,
    multi_conv_receptive_field_center,
    multi_conv_receptive_field_size,
)



class SincNetPool(nn.Module):
    def __init__(self, sample_rate: int, stride: int, ksize: int = 251):
        super().__init__()

        if sample_rate != 16000:
            raise NotImplementedError("SincNet only supports 16kHz audio for now.")
            # TODO: add support for other sample rate. it should be enough to multiply
            # kernel_size by (sample_rate / 16000). but this needs to be double-checked.

        self.sample_rate = sample_rate
        self.stride = stride

        # 오디오 신호의 정규화를 위한 1D 인스턴스 정규화 레이어
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        # Convolutional layers, pooling layers, and normalization layers
        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        # 첫 번째 레이어: SincNet의 핵심인 Sinc 필터를 사용한 컨볼루션 레이어
        self.ksize = 321 # 20.6ms
        self.ksize = ksize # 251: 15.6ms
        self.conv1d.append(
            asteroid_filterbanks.Encoder(
                asteroid_filterbanks.ParamSincFB( # Filterbank
                    n_filters=80,
                    kernel_size=self.ksize,
                    stride=stride,
                    sample_rate=sample_rate,
                    min_low_hz=50,
                    min_band_hz=50,
                )
            )
        )
        # Max pooling layer to reduce the dimensionality
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        # Normalization layer to stabilize the learning process
        self.norm1d.append(nn.InstanceNorm1d(num_features=80, affine=True))

        # 두 번째 컨볼루션 레이어
        self.conv1d.append(nn.Conv1d(in_channels=80, out_channels=60, kernel_size=5, stride=1))
        self.pool1d.append(nn.MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(num_features=60, affine=True))

        self.conv1d.append(nn.Conv1d(in_channels=60, out_channels=60, kernel_size=5, stride=1))
        self.pool1d.append(nn.MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(num_features=60, affine=True))

        self.model_cfg = {
            "kernel_size": [self.ksize, 3, 5, 3, 5, 3],
            "stride": [self.stride, 3, 1, 3, 1, 3],# 총 축소율: 약 3 × 3 × 3 = 27배
            "padding": [0, 0, 0, 0, 0, 0],
            "dilation": [1, 1, 1, 1, 1, 1],
        }
    @lru_cache
    def num_frames(self, n_audio_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """
        return multi_conv_num_frames(
            n_audio_samples, **self.model_cfg
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
        return multi_conv_receptive_field_size(
            num_frames, **self.model_cfg
        )

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
        return multi_conv_receptive_field_center(
            frame, **self.model_cfg
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)
        """
        # 입력 오디오 신호를 정규화
        outputs = self.wav_norm1d(waveforms)

        for c, (conv1d, pool1d, norm1d) in enumerate(
            zip(self.conv1d, self.pool1d, self.norm1d)
        ):
            outputs = conv1d(outputs)

            # https://github.com/mravanelli/SincNet/issues/4
            # 첫 번째 레이어의 출력은 절대값을 취함
            if c == 0:
                outputs = torch.abs(outputs)

            # 풀링과 정규화, 그리고 활성화 함수 적용
            outputs = F.leaky_relu(norm1d(pool1d(outputs)))

        return outputs

    def receptive_field(self):
        size = self.receptive_field_size(num_frames=1)
        step = (
            self.receptive_field_size(num_frames=2) - size
        )
        start = (
            self.receptive_field_center(frame=0) - (size - 1) / 2
        )
        return start, size, step


'''
# 입력: 2.25초(36,000 샘플), 목표 출력: 20~24 프레임
stride = 56         # 첫 레이어의 스트라이드, 3.5ms
stride = 54         # 첫 레이어의 스트라이드, 3.5ms
kernel_size = 321   # 커널 크기 (약 20ms 컨텍스트)

36,000 ÷ 56 = 642.857 ≈ 643
36,000 ÷ 54 = 666.666 ≈ 667
643 ÷ 3 ÷ 3 ÷ 3 ≈ 23.8 (24frame)
'''
if __name__ == "__main__":
    stride = 10 # def
    ksize = 251

    stride = 56
    ksize = 321
    print(f'\n\ntest stride={stride}, ksize={ksize}')
    sincnet = SincNetPool(sample_rate=16000, stride=stride, ksize=ksize)

    input_sec = 2.25
    n_frame = sincnet.num_frames(n_audio_samples=int(16000*input_sec))

    print(f'{n_frame=}')
    print('receptive_field_size=',sincnet.receptive_field_size(num_frames=1))
    print('receptive_field_center=',sincnet.receptive_field_center())
    print('receptive_field=',sincnet.receptive_field())
    print(f"frame ms={input_sec/n_frame:.3f}")

