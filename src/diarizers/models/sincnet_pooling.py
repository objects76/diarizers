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

'''
# 입력: 2.25초(36,000 샘플), 목표 출력: 20~24 프레임
stride = 56         # 첫 레이어의 스트라이드, 3.5ms
stride = 54         # 첫 레이어의 스트라이드, 3.37ms
kernel_size = 321   # 커널 크기 (약 20ms 컨텍스트)

36,000 ÷ 56 = 642.857 ≈ 643
36,000 ÷ 54 = 666.666 ≈ 667
643 ÷ 3 ÷ 3 ÷ 3 ≈ 23.8 (24frame)
'''

class SincNetPool(nn.Module):
    def __init__(self, sample_rate: int = 16000,
                 stride: int= 10, ksize: int = 251,
                 frame_sec: float = 0.10384*0):
        super().__init__()

        assert sample_rate == 16000, "SincNetPool only supports 16kHz audio for now."
        self.sample_rate = sample_rate
        self.stride = stride

        # to use pretrained pyan weight and make output granularity bigger.
        self.post_pool = None

        # 오디오 신호의 정규화를 위한 1D 인스턴스 정규화 레이어
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        # Convolutional layers, pooling layers, and normalization layers
        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        # 첫 번째 레이어: SincNet의 핵심인 Sinc 필터를 사용한 컨볼루션 레이어
        self.ksize = 321 # 20.6ms
        self.ksize = ksize # 251: 15.6ms
        t_sinc_fb = asteroid_filterbanks.ParamSincFB( # Filterbank
                    n_filters=80,
                    kernel_size=self.ksize,
                    stride=stride,
                    sample_rate=sample_rate,
                    min_low_hz=50,
                    min_band_hz=50,
                )

        self.conv1d.append( asteroid_filterbanks.Encoder(t_sinc_fb))
        # Max pooling layer to reduce the dimensionality
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        # Normalization layer to stabilize the learning process
        self.norm1d.append(nn.InstanceNorm1d(num_features=80, affine=True))

        # 두 번째 컨볼루션 레이어
        self.conv1d.append(nn.Conv1d(in_channels=80, out_channels=60, kernel_size=5, stride=1)) # t
        self.pool1d.append(nn.MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(num_features=60, affine=True))

        self.conv1d.append(nn.Conv1d(in_channels=60, out_channels=60, kernel_size=5, stride=1)) # t
        self.pool1d.append(nn.MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(num_features=60, affine=True))

        if frame_sec > 0:
            assert ksize == 251 and stride == 10, "SincNetPool only supports 251, 10 for now."
            # 기존 스텝 크기 계산 (대략 0.017307692초)
            base_step = 0.017307692

            # 목표 스텝 크기(예: 0.102초)에 맞는 풀링 커널 크기 계산
            pool_kernel: int = math.ceil(frame_sec / base_step)
            pool_stride: int = pool_kernel

            self.post_pool = (
                nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, ceil_mode=True)
            )
            print(f"SincNetPool: {pool_kernel=}/{frame_sec / base_step:.3f}, {pool_stride=}, {frame_sec=}")


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

    def _receptive_field_size(self, num_frames: int = 1) -> int:
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

    def _receptive_field_center(self, frame: int = 0) -> int:
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

        if self.post_pool is not None:
            outputs = self.post_pool(outputs)

        return outputs

    def receptive_field(self):
        size = self._receptive_field_size(num_frames=1)
        step = (
            self._receptive_field_size(num_frames=2) - size
        )
        start = (
            self._receptive_field_center(frame=0) - (size - 1) / 2
        )

        # post_pool이 있는 경우 리셉티브 필드 정보를 조정
        if self.post_pool is not None:
            # post_pool이 있을 때 고정 커널 크기 6을 사용
            pool_kernel = self.post_pool.kernel_size[0] # type: ignore
            pool_stride = pool_kernel
            print(f"post_pool: {pool_kernel=}, {pool_stride=}")

            # post_pool로 인한 스텝 크기 조정
            step = step * pool_stride

            # post_pool로 인한 리셉티브 필드 크기 확장
            size = size + (pool_kernel - 1) * step / pool_stride
            # 시작 지점은 변하지 않음 (첫 번째 프레임의 중심점은 동일)

        return start, size, step



if __name__ == "__main__":

    def test_sincnet_forward(sincnet=None):
        if sincnet is None:
            ksize, stride = 251, 10 # def
            sincnet = SincNetPool(stride=stride, ksize=ksize, frame_sec=0.10384)

        # Calculate the number of samples for the given duration
        input_duration_sec = 2.25
        num_samples = int(16000 * input_duration_sec)

        # Create a random input tensor simulating a batch of audio waveforms
        # Shape: (batch_size, num_channels, num_samples)
        batch_size = 4
        num_channels = 1
        input_waveforms = torch.randn(batch_size, num_channels, num_samples)

        # Initialize the SincNetPool with the given stride and kernel size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sincnet.to(device)

        # Move input tensor to the same device as the model
        input_waveforms = input_waveforms.to(device)

        # Perform the forward pass
        output = sincnet(input_waveforms)

        # Print the shape of the output to verify the forward pass
        print(f"Output shape: {output.shape}")

    # test_sincnet_forward()

    def receptive_field_test():
        ksize, stride = 251, 10 # def
        # ksize, stride = 321, 56
        frame_sec = 0.10384
        print(f'\n\ntest stride={stride}, ksize={ksize}')
        sincnet = SincNetPool(stride=stride, ksize=ksize, frame_sec=frame_sec)

        input_sec = 2.25
        n_frame = sincnet.num_frames(n_audio_samples=int(16000*input_sec))

        print(f'{n_frame=}')
        print('receptive_field_size=',sincnet._receptive_field_size(num_frames=1))
        print('receptive_field_center=',sincnet._receptive_field_center())
        print('receptive_field=',sincnet.receptive_field())
        print(f"frame ms={input_sec/n_frame:.3f}")

        test_sincnet_forward(sincnet)

    receptive_field_test()
