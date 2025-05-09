# Adapted from https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/tasks/segmentation/speaker_diarization.py
# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math

import numpy as np
import torch

from ..models import SegmentationModel
from pyannote.core import SlidingWindow # jjkim

class Preprocess:
    """Converts a HF dataset with the following features:
        - "audio": Audio feature.
        - "speakers": The list of audio speakers, with their order of appearance.
        - "timestamps_start": A list of timestamps indicating the start of each speaker segment.
        - "timestamps_end": A list of timestamps indicating the end of each speaker segment.
    to a preprocessed dataset ready to be used with the HF Trainer.
    """

    def __init__(
        self,
        config,
        n_receptive_field_frame,
    ):
        """Preprocess init method.
        Takes as input the dataset to process and the model to perform training with.
        The preprocessing is done to fit the hyperparameters of the model.
        Args:
            input_dataset (dataset): Hugging Face Speaker Diarization dataset
            model (SegmentationModel): A SegmentationModel from the diarizers library.
        """
        self.chunk_duration = config.chunk_duration
        self.max_speakers_per_frame = config.max_speakers_per_frame
        self.max_speakers_per_chunk = config.max_speakers_per_chunk
        self.min_duration = config.min_duration
        self.warm_up = config.warm_up

        self.sample_rate = config.sample_rate
        model = SegmentationModel(config).to_pyannote_model()

        # Get the number of frames associated to a chunk:
        _, self.num_frames_per_chunk, n_speakers = model(
            torch.rand((1, int(self.chunk_duration * self.sample_rate)))
        ).shape

        self.receptive_field_step = model.receptive_field.step
        self.receptive_field_duration = 0.5 * model.receptive_field.duration
        self.post_pool_size = model.post_pool.kernel_size[0] if model.post_pool else 0

        if n_receptive_field_frame > 0:
            self.num_frames_per_chunk = n_receptive_field_frame
        print(f"{self.num_frames_per_chunk=} for {self.chunk_duration}")

    def get_labels_in_file(self, file):
        """Get speakers in file.
        Args:
            file (_type_): dataset row from input dataset.

        Returns:
            file_labels (list): a list of all speakers in the audio file.
        """

        file_labels = []
        for i in range(len(file["speakers"][0])):
            if file["speakers"][0][i] not in file_labels:
                file_labels.append(file["speakers"][0][i])

        return file_labels

    def get_segments_in_file(self, file, labels) -> np.ndarray:
        """Get segments in file.

        Args:
            file (_type_): dataset row from input dataset.
            labels (_type_):  a list of all speakers in the audio file.

        Returns:
            annotations (numpy array): _description_
        """

        file_annotations = []

        for i in range(len(file["timestamps_start"][0])):
            start_segment = file["timestamps_start"][0][i]
            end_segment = file["timestamps_end"][0][i]
            label = labels.index(file["speakers"][0][i])
            file_annotations.append((start_segment, end_segment, label))

        dtype = [("start", "<f4"), ("end", "<f4"), ("labels", "i1")]

        # [(start, end, label), ...]
        annotations = np.array(file_annotations, dtype)
        return annotations

    def get_chunk(self, file, start_time, min_speaker,
                  debug=False) -> tuple[np.ndarray, np.ndarray, list] | None:
        """Method used to get an audio chunk from an audio file given a start_time.

        Args:
            file (dict): dataset row containing the "audio" feature.
            start_time (float): start time (in seconds) of the audio_chunk to extract.

        Returns:
            tuple of (waveform, y, labels) or None:
                waveform (np.ndarray): audio chunk
                y (np.ndarray): target array.
                labels (list): list of speakers in chunk.
                None: if the chunk doesn't meet requirements.
        """
        sample_rate = file["audio"][0]["sampling_rate"]

        assert sample_rate == self.sample_rate

        end_time = start_time + self.chunk_duration
        start_frame = math.floor(start_time * sample_rate)
        num_frames_waveform = math.floor(self.chunk_duration * sample_rate)
        end_frame = start_frame + num_frames_waveform

        waveform:np.ndarray = file["audio"][0]["array"][start_frame:end_frame]
        if len(waveform) < num_frames_waveform:
            padding_length = num_frames_waveform - len(waveform)
            waveform = np.pad(waveform, (0, padding_length), 'constant', constant_values=0)

        assert len(waveform) == num_frames_waveform

        labels = self.get_labels_in_file(file)
        # print('get_chunk.labels:', labels)

        file_segments = self.get_segments_in_file(file, labels); del labels
        # print('file_segments:', file_segments)

        # overlapped segments.
        chunk_segments = file_segments[
            (file_segments["start"] < end_time) & (file_segments["end"] > start_time)]
        # print('chunk_segments:', chunk_segments)

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_segments["labels"]))
        num_labels = len(labels)
        if num_labels < min_speaker:
            return None

        # compute frame resolution:
        # resolution = self.chunk_duration / self.num_frames_per_chunk


        # discretize chunk annotations at model output resolution
        step = self.receptive_field_step
        half = 0.5 * self.receptive_field_duration

        # 스텝 정보 출력 (디버깅 목적)
        # DEV이 True일 때만 출력하도록 수정
        if debug:
            print(f"Receptive field: step={step:.6f}s, duration={self.receptive_field_duration:.6f}s")
            if self.post_pool_size:
                print(f"Using post_pool with kernel_size={self.post_pool_size}")

        # discretize chunk annotations at model output resolution
        start1 = np.maximum(chunk_segments["start"], start_time) - start_time - half
        start_idx = np.maximum(0, np.round(start1 / step)).astype(int)
        # start_idx = np.floor(start / resolution).astype(int)

        end1 = np.minimum(chunk_segments["end"], end_time) - start_time - half
        end_idx = np.round(end1 / step).astype(int)
        # end_idx = np.ceil(end / resolution).astype(int)

        # 배열 크기 제한 확인 및 조정 (디버깅 용)
        if debug and max(end_idx) >= self.num_frames_per_chunk:
            print(f"Warning: end_idx {max(end_idx)} exceeds num_frames_per_chunk {self.num_frames_per_chunk}")
            end_idx = np.minimum(end_idx, self.num_frames_per_chunk - 1)

        # initial frame-level targets
        y = np.zeros((self.num_frames_per_chunk, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}
        if debug: print(mapping)

        for start, end, label in zip(start_idx, end_idx, chunk_segments["labels"]):
            mapped_label = mapping[label]
            y[start : end + 1, mapped_label] = 1

        #+ jjkim
        if self.post_pool_size:
            n_combine = self.post_pool_size
            n_label = n_combine * math.ceil(y.shape[0] / n_combine)
            ypadded = np.pad(y,
                       # 0 rows before, 10 rows after, 0 cols before or after
                       pad_width=((0, n_combine-1), (0,  0)),
                       mode='edge'
            )
            blocks = ypadded[:n_label].reshape(-1, n_combine, ypadded.shape[1])
            avg = blocks.mean(axis=1)
            if debug: print('merged label:', str(avg).replace('\n', ','))

            # result = np.round(avg).astype(np.uint8)
            result = np.where(avg >= 0.4, 1, 0).astype(np.uint8) # (avg > 0.7) ? 1:0
            if debug: print(f'{n_combine=}, {y.shape=}, {ypadded.shape=}, {result.shape=}')
            y = result # (22,2)

            # manual filtering
            # [1 0] ([0 1] or [0 0]) - removable: this case will be removed by chunked embedding?
            if np.array_equal(y[0], [1, 0]) and y[1][0] != 1:
                return None

            # if only 1 speaker after blocks.mean
            if len(np.unique(y, axis=0)) == 1:
                return None
        #- jjkim

        return waveform, y, labels

    def get_start_positions(self, file, overlap, random=False):
        """Get start positions from the audio_chunks in the input audio file.

        Args:
            file (dict): dataset row containing the "audio" feature.
            overlap (float, optional):  Overlap between successive start positions.
            random (bool, optional):  Whether or not to randomly select chunks in the audio file. Defaults to False.

        Returns:
            start_positions: Numpy array containing the start positions of the audio chunks in file.
        """

        sample_rate = file["audio"][0]["sampling_rate"]

        assert sample_rate == self.sample_rate
        assert overlap < 1, f"overlap must be less than 1"
        assert self.chunk_duration > 0.0, f"self.chunk_duration must be greater than 0.0"

        step = 1 - overlap
        file_duration = len(file["audio"][0]["array"]) / sample_rate
        start_positions = np.arange(0, file_duration - self.chunk_duration, step = step)

        if random:
            nb_samples = int(file_duration / self.chunk_duration) # number of batch_samples
            start_positions = np.random.uniform(0, file_duration, nb_samples)

        return start_positions

    def apply_eval(self, file):
        DEV = False

        new_batch = {"waveforms": [], "labels": [], "nb_speakers": []}

        start_positions = [0]
        if DEV: print(f'{start_positions=}')

        for start_time in start_positions:
            result = self.get_chunk(file, start_time, min_speaker=2)
            if result is None: continue

            waveform, target, label = result
            new_batch["waveforms"].append(waveform)
            new_batch["labels"].append(target)
            new_batch["nb_speakers"].append(label)

        return new_batch

    def __call__(self, file, random=False, overlap=0.0, min_person=2):
        """Chunk an audio file into short segments of duration self.chunk_duration

        Args:
            file (dict): dataset row containing the "audio" feature.
            random (bool, optional): Whether or not to randomly select chunks in the audio file. Defaults to False.
            overlap (float, optional):  Overlap between successive chunks. Defaults to 0.0.

        Returns:
            new_batch: new batch containing for each chunk the corresponding waveform, labels and number of speakers.
        """
        DEV = False

        new_batch = {"waveforms": [], "labels": [], "nb_speakers": []}

        if random:
            start_positions = self.get_start_positions(file, overlap, random=True)
        else:
            start_positions = self.get_start_positions(file, overlap)
            if DEV: print(f'{start_positions=}')

        for start_time in start_positions:
            result = self.get_chunk(file, start_time, min_speaker=min_person)
            if result is None: continue

            waveform, target, label = result
            new_batch["waveforms"].append(waveform)
            new_batch["labels"].append(target)
            new_batch["nb_speakers"].append(label)

        return new_batch