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

from diarizers.models.pyannet import PyanNet_nn
import numpy as np
import torch
from datasets.formatting.formatting import LazyRow, LazyBatch

SR = 16_000
CHUNK_DURATION = 2.25

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
        *,
        config = None,
        num_frames_per_chunk:int = 0,
        receptive_field_step:float = 0, receptive_field_duration:float = 0
    ):
        """Preprocess init method.
        Takes as input the dataset to process and the model to perform training with.
        The preprocessing is done to fit the hyperparameters of the model.
        Args:
            input_dataset (dataset): Hugging Face Speaker Diarization dataset
            model (SegmentationModel): A SegmentationModel from the diarizers library.
        """
#         if config is not None:
#             from diarizers.models import SegmentationModel, SegmentationModelConfig
#             assert config.chunk_duration == CHUNK_DURATION
#             # CHUNK_DURATION = config.chunk_duration # 2.25
#             # self.max_speakers_per_frame = config.max_speakers_per_frame # 2
#             # self.max_speakers_per_chunk = config.max_speakers_per_chunk # 2
#             # self.min_duration = config.min_duration # 0.0
#             # self.warm_up = config.warm_up # (0,0)
#
#             assert SR == config.sample_rate # 16000
#
#             # 올바른 SegmentationModelConfig로 모델 초기화
#             model_config = SegmentationModelConfig(
#                 chunk_duration=CHUNK_DURATION,
#                 max_speakers_per_frame=config.max_speakers_per_frame, # 2
#                 max_speakers_per_chunk=config.max_speakers_per_chunk, # 2
#                 min_duration=config.min_duration, # 0
#                 warm_up=config.warm_up,
#                 sincnet=config.sincnet, # {'ksize': 251, 'stride': 10}
#             )
#             model:PyanNet_nn = SegmentationModel(config=model_config).pyan_nn
#
#             # Get the number of frames associated to a chunk:
#             _, num_frames_per_chunk, n_speakers = model(
#                 torch.rand((1, int(CHUNK_DURATION * SR)))
#             ).shape
#
#             # receptive field
#             receptive_field_step = model.receptive_field.step
#             receptive_field_duration = 0.5 * model.receptive_field.duration
#             # self.post_pool_size = model.post_pool.kernel_size[0] if model.post_pool else 0
#             print(f"{receptive_field_step=}")
#             print(f"{receptive_field_duration=}")
#             print(f"{num_frames_per_chunk=} for {CHUNK_DURATION}")
#             del model

        assert num_frames_per_chunk > 0, f"invalid {num_frames_per_chunk}"
        assert receptive_field_step > 0, f"invalid {receptive_field_step}"
        assert receptive_field_duration > 0, f"invalid {receptive_field_duration}"
        self.num_frames_per_chunk = num_frames_per_chunk
        self.receptive_field_step = receptive_field_step
        self.receptive_field_duration = receptive_field_duration

    # @staticmethod
    # def get_labels_in_file(file) -> list[str]:
    #     file_labels: list[str] = []
    #     for i in range(len(file["speakers"][0])):
    #         if file["speakers"][0][i] not in file_labels:
    #             file_labels.append(file["speakers"][0][i])
    #     return file_labels

    @staticmethod
    def get_segments_in_file(file, labels) -> np.ndarray:
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
            label = labels.index(file["speakers"][0][i]) # 0 or 1
            file_annotations.append((start_segment, end_segment, label))

        dtype = [("start", "<f4"), ("end", "<f4"), ("label", "i1")]

        # [(start, end, label), ...]
        annotations = np.array(file_annotations, dtype)
        return annotations

    def get_chunk(self,
                  audio_array, sampling_rate,
                  timestamp_start, timestamp_end,
                  speakers,
                  start_time, min_speaker, debug=False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Method used to get an audio chunk from an audio file given a start_time.

        Args:
            audio_array (np.ndarray): raw audio data
            sampling_rate (int): sampling rate of the audio
            timestamp_start (np.ndarray): start timestamps for each segment
            timestamp_end (np.ndarray): end timestamps for each segment
            speakers (np.ndarray): speaker labels for each segment
            start_time (float): start time (in seconds) of the audio_chunk to extract.
            min_speaker (int): minimum number of speakers required
            debug (bool): debug mode

        Returns:
            tuple of (waveform, y, labels) or None:
                waveform (np.ndarray): audio chunk
                y (np.ndarray): target array.
                labels (list): list of speakers in chunk.
                None: if the chunk doesn't meet requirements.
        """
        assert sampling_rate == SR, f'assert: {sampling_rate} == {SR}'

        end_time = start_time + CHUNK_DURATION
        start_frame = math.floor(start_time * SR)
        num_frames_waveform = math.floor(CHUNK_DURATION * SR)
        end_frame = start_frame + num_frames_waveform

        waveform:np.ndarray = np.array(
            audio_array[start_frame:end_frame],
            dtype=np.float32 )
        if len(waveform) < num_frames_waveform:
            padding_length = num_frames_waveform - len(waveform)
            waveform = np.pad(waveform, (0, padding_length), 'constant', constant_values=0)

        assert len(waveform) == num_frames_waveform, f"assert: {len(waveform)} == {num_frames_waveform}"

        # Convert element arrays to DataFrame for processing
        import pandas as pd
        file_segments = pd.DataFrame({
            'start': timestamp_start,
            'end': timestamp_end,
            'label': speakers
        })
        if debug: print('file_segments:', file_segments)

        # overlapped segments.
        chunk_segments = file_segments[
            (start_time < file_segments["end"]) & (file_segments["start"] < end_time)
            ]
        if debug: print('chunk_segments:', chunk_segments)

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_segments["label"]))
        if debug: print('labels:', labels)

        num_labels = len(labels)
        if num_labels < min_speaker:
            return None

        # compute frame resolution:
        # resolution = CHUNK_DURATION / self.num_frames_per_chunk

        # discretize chunk annotations at model output resolution
        step = self.receptive_field_step
        half = 0.5 * self.receptive_field_duration

        # discretize chunk annotations at model output resolution
        start = np.maximum(chunk_segments["start"], start_time) - (start_time + half)
        start_idx = np.maximum(0, np.round(start / step - 0.2)).astype(int) # round down at 0.7
        # start_idx = np.floor(start / resolution).astype(int)

        end = np.minimum(chunk_segments["end"], end_time) - (start_time + half)
        end_idx = np.round(end / step).astype(int)
        # end_idx = np.ceil(end / resolution).astype(int)

        # 배열 크기 제한 확인 및 조정 (디버깅 용)
        if debug and max(end_idx) >= self.num_frames_per_chunk:
            print(f"Warning: end_idx {max(end_idx)} exceeds num_frames_per_chunk {self.num_frames_per_chunk}")
            end_idx = np.minimum(end_idx, self.num_frames_per_chunk - 1)

        # initial frame-level targets
        y = np.zeros((self.num_frames_per_chunk, num_labels), dtype=np.uint8)

        # map labels to indices: label -> [0,1,...]
        mapping = {label: idx for idx, label in enumerate(labels)}
        if debug: print(mapping)

        for i_start, i_end, label in zip(start_idx, end_idx, chunk_segments["label"]):
            label_pos = mapping[label]
            y[i_start : i_end + 1, label_pos] = 1

        return waveform, y, list(range(num_labels))
        return waveform, y, np.array(list(range(num_labels)), dtype=np.int32)

    def get_start_positions(self, file, overlap, random=False):
        """Get start positions from the audio_chunks in the input audio file.

        Args:
            file (dict): dataset row containing the "audio" feature.
            overlap (float, optional):  Overlap between successive start positions.
            random (bool, optional):  Whether or not to randomly select chunks in the audio file. Defaults to False.

        Returns:
            start_positions: Numpy array containing the start positions of the audio chunks in file.
        """

        assert file["audio"][0]["sampling_rate"] == SR
        assert overlap < 1, f"overlap must be less than 1"

        step = 1 - overlap
        file_duration = len(file["audio"][0]["array"]) / SR
        start_positions = np.arange(0, file_duration - CHUNK_DURATION, step = step)

        if random:
            nb_samples = int(file_duration / CHUNK_DURATION) # number of batch_samples
            start_positions = np.random.uniform(0, file_duration, nb_samples)

        return start_positions

    def apply_eval(self, batch:LazyBatch|LazyRow):
        new_batch = {"waveforms": [], "labels": [], "idx_speakers": []}

        audio = batch["audio"]
        for idx in range(len(audio)):
            # Extract element data from sample
            timestamp_start = batch["timestamps_start"][idx]
            timestamp_end = batch["timestamps_end"][idx]
            speakers = batch["speakers"][idx]

            result = self.get_chunk(
                audio_array=audio[idx]["array"],
                sampling_rate=audio[idx]["sampling_rate"],
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                speakers=speakers,
                start_time=0,
                min_speaker=2
            )
            if result:
                waveform, target, label = result
                new_batch["waveforms"].append(waveform)
                new_batch["labels"].append(target)
                new_batch["idx_speakers"].append(label)
        return new_batch

    def __call__(self, file, random=False, overlap=0.0, min_person=2):
        """Chunk an audio file into short segments of duration CHUNK_DURATION

        Args:
            file (dict): dataset row containing the "audio" feature.
            random (bool, optional): Whether or not to randomly select chunks in the audio file. Defaults to False.
            overlap (float, optional):  Overlap between successive chunks. Defaults to 0.0.

        Returns:
            new_batch: new batch containing for each chunk the corresponding waveform, labels and number of speakers.
        """
        DEV = False

        new_batch = {"waveforms": [], "labels": [], "idx_speakers": []}

        if random:
            start_positions = self.get_start_positions(file, overlap, random=True)
        else:
            start_positions = self.get_start_positions(file, overlap)
            if DEV: print(f'{start_positions=}')

        for start_time in start_positions:
            # Extract element data from file
            audio_array = file["audio"][0]["array"]
            sampling_rate = file["audio"][0]["sampling_rate"]
            timestamp_start = file["timestamps_start"][0] if "timestamps_start" in file else []
            timestamp_end = file["timestamps_end"][0] if "timestamps_end" in file else []
            speakers = file["speakers"][0] if "speakers" in file else []

            result = self.get_chunk(
                audio_array=audio_array,
                sampling_rate=sampling_rate,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                speakers=speakers,
                start_time=start_time,
                min_speaker=min_person
            )
            if result is None: continue

            waveform, target, label = result
            new_batch["waveforms"].append(waveform)
            new_batch["labels"].append(target)
            new_batch["idx_speakers"].append(label)

        return new_batch

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class PreprocessConfig:
        chunk_duration: float = 2.25
        max_speakers_per_frame: int = 2
        max_speakers_per_chunk: int = 2
        min_duration: float = 0.0
        warm_up: float = 0.0
        sample_rate: int = 16000

    processor = Preprocess(PreprocessConfig())

    def save_test_chunk(file: dict, start_time: float, min_speaker: int, output_dir: str) -> None:
        # Get the chunk using the processor
        # Extract element data from file
        audio_array = file["audio"][0]["array"]
        sampling_rate = file["audio"][0]["sampling_rate"]
        timestamp_start = file["timestamps_start"][0] if "timestamps_start" in file else []
        timestamp_end = file["timestamps_end"][0] if "timestamps_end" in file else []
        speakers = file["speakers"][0] if "speakers" in file else []

        result = processor.get_chunk(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            speakers=speakers,
            start_time=start_time,
            min_speaker=min_speaker
        )
        if result is None:
            print("No valid chunk found.")
            return

        waveform, target, label = result

        # Prepare data to save
        chunk_data = {
            "waveform": waveform.tolist(),
            "target": target.tolist(),
            "label": label
        }
        print(f"{waveform.shape=}, {target.shape=}, {label=}")
        target_str = str(target).replace('\n', '')
        print(f"{target_str=}")

    test_file = {
        "audio": [{"sampling_rate": 16000, "array": np.random.rand(int(16000 * 2.25))}],  # 2.25 seconds of random audio
        "speakers": [[0, 1]],
        "timestamps_start": [[0.0, 1.120]],
        "timestamps_end": [[1.125, 2.25]]
    }
    save_test_chunk(test_file, start_time=0, min_speaker=2, output_dir="test_chunks")