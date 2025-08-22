import numpy as np
import torch
from pyannote.audio.torchmetrics import (DiarizationErrorRate, FalseAlarmRate,
                                         MissedDetectionRate,
                                         SpeakerConfusionRate)
from pyannote.audio.utils.powerset import Powerset


class Metrics:
    """Metric class used by the HF trainer to compute speaker diarization metrics."""

    def __init__(self, specifications) -> None:
        """init method

        Args:
            specifications (_type_): specifications attribute from a SegmentationModel.
        """
        self.powerset = specifications.powerset
        self.classes = specifications.classes
        self.powerset_max_classes = specifications.powerset_max_classes

        self.model_powerset = Powerset(
            len(self.classes),
            self.powerset_max_classes,
        )

        self.metrics = {
            "der": DiarizationErrorRate(0.5),
            "confusion": SpeakerConfusionRate(0.5),
            "missed_detection": MissedDetectionRate(0.5),
            "false_alarm": FalseAlarmRate(0.5),
        }

    def __call__(self, eval_pred):

        logits, labels = eval_pred

        if self.powerset:
            predictions = self.model_powerset.to_multilabel(torch.tensor(logits))
        else:
            predictions = torch.tensor(logits)

        labels = torch.tensor(labels)

        predictions = torch.transpose(predictions, 1, 2)
        labels = torch.transpose(labels, 1, 2)

        metrics = {"der": 0, "false_alarm": 0, "missed_detection": 0, "confusion": 0}

        metrics["der"] += self.metrics["der"](predictions, labels).cpu().numpy()
        metrics["false_alarm"] += self.metrics["false_alarm"](predictions, labels).cpu().numpy()
        metrics["missed_detection"] += self.metrics["missed_detection"](predictions, labels).cpu().numpy()
        metrics["confusion"] += self.metrics["confusion"](predictions, labels).cpu().numpy()

        return metrics


class DataCollator:
    """Data collator that will dynamically pad the target labels to have max_speakers_per_chunk"""

    def __init__(self, max_speakers_per_chunk) -> None:
        self.max_speakers_per_chunk = max_speakers_per_chunk

    def __call__(self, features):
        """_summary_

        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """

        batch = {}

        speakers = [f["idx_speakers"] for f in features]
        labels = [f["labels"] for f in features]

        batch["labels"] = self.pad_targets(labels, speakers)

        batch["waveforms"] = torch.stack([f["waveforms"] for f in features])

        return batch

    def pad_targets(self, labels, speakers):
        """
        labels:
        speakers:

        Returns:
            _type_:
                Collated target tensor of shape (num_frames, self.max_speakers_per_chunk)
                If one chunk has more than max_speakers_per_chunk speakers, we keep
                the max_speakers_per_chunk most talkative ones. If it has less, we pad with
                zeros (artificial inactive speakers).
        """

        targets = []

        for i in range(len(labels)):
            label = speakers[i]
            target = labels[i].numpy()
            num_speakers = len(label)

            if num_speakers > self.max_speakers_per_chunk:
                indices = np.argsort(-np.sum(target, axis=0), axis=0)
                target = target[:, indices[: self.max_speakers_per_chunk]]

            elif num_speakers < self.max_speakers_per_chunk:
                target = np.pad(
                    target,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )

            targets.append(target)

        return torch.from_numpy(np.stack(targets))
