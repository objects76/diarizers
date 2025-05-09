import copy
from typing import Optional
from pathlib import Path

import torch
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.utils.loss import binary_cross_entropy, nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.powerset import Powerset
import transformers

from diarizers.models.pyannet import PyanNet_nn
from diarizers.models.pyannet import PyanNet

from functools import cached_property
from pyannote.core import SlidingWindow

class SegmentationModelConfig(transformers.PretrainedConfig):
    """Config class associated with SegmentationModel model."""

    model_type = "pyannet"

    def __init__(
        self,
        chunk_duration=10,
        max_speakers_per_frame=2,
        max_speakers_per_chunk=3,
        min_duration=None,
        warm_up=(0.0, 0.0),
        weigh_by_cardinality=False,
        sincnet=None,
        **kwargs,
    ):
        """Init method for the
        Args:
            chunk_duration : float, optional
                    Chunks duration processed by the model. Defaults to 10s.
            max_speakers_per_chunk : int, optional
                Maximum number of speakers per chunk.
            max_speakers_per_frame : int, optional
                Maximum number of (overlapping) speakers per frame.
                Setting this value to 1 or more enables `powerset multi-class` training.
                Default behavior is to use `multi-label` training.
            weigh_by_cardinality: bool, optional
                Weigh each powerset classes by the size of the corresponding speaker set.
                In other words, {0, 1} powerset class weight is 2x bigger than that of {0}
                or {1} powerset classes. Note that empty (non-speech) powerset class is
                assigned the same weight as mono-speaker classes. Defaults to False (i.e. use
                same weight for every class). Has no effect with `multi-label` training.
            min_duration : float, optional
                Sample training chunks duration uniformely between `min_duration`
                and `duration`. Defaults to `duration` (i.e. fixed length chunks).
            warm_up : float or (float, float), optional
                Use that many seconds on the left- and rightmost parts of each chunk
                to warm up the model. While the model does process those left- and right-most
                parts, only the remaining central part of each chunk is used for computing the
                loss during training, and for aggregating scores during inference.
                Defaults to 0. (i.e. no warm-up).
        """
        super().__init__(**kwargs)
        self.chunk_duration = chunk_duration
        self.max_speakers_per_frame = max_speakers_per_frame
        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.min_duration = min_duration
        self.warm_up = warm_up
        self.weigh_by_cardinality = weigh_by_cardinality
        # For now, the model handles only 16000 Hz sampling rate
        self.sample_rate = 16000

        # 추가 코드
        self.sincnet = sincnet or {
            "ksize": 251,
            "stride": 10,
            "frame_sec": 0.10384
        }


class SegmentationModel(transformers.PreTrainedModel):
    """
    Wrapper class for the PyanNet segmentation model used in pyannote.
    Inherits from Pretrained model to be compatible with the HF Trainer.
    Can be used to train segmentation models to be used for the "SpeakerDiarisation Task" in pyannote.
    """

    # config_class = SegmentationModelConfig

    def __init__(
        self,
        config=SegmentationModelConfig(),
    ):
        """init method
        Args:
            config (SegmentationModelConfig): instance of SegmentationModelConfig.
        """

        super().__init__(config)


        self.weigh_by_cardinality = config.weigh_by_cardinality
        self.max_speakers_per_frame = config.max_speakers_per_frame
        self.chunk_duration = config.chunk_duration
        self.min_duration = config.min_duration
        self.warm_up = config.warm_up
        self.max_speakers_per_chunk = config.max_speakers_per_chunk

        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION \
                if self.max_speakers_per_frame is None
                else Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.chunk_duration,
            min_duration=self.min_duration,
            warm_up=self.warm_up,
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
            permutation_invariant=True,
        )

        self.pyan_nn = PyanNet_nn(sincnet=config.sincnet)
        self.pyan_nn.specifications = self.specifications
        self.pyan_nn.build()
        self.setup_loss_func()

    def forward(self,
                waveforms: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                nb_speakers: Optional[int] = None) -> dict:
        """foward pass of the Pretrained Model.

        Args:
            waveforms (torch.tensor): _description_
            labels (_type_, optional): _description_. Defaults to None.
            nb_speakers (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        prediction = self.pyan_nn(waveforms.unsqueeze(1))
        batch_size, num_frames, _ = prediction.shape

        if labels is not None:
            weight = torch.ones(batch_size, num_frames, 1, device=waveforms.device)
            warm_up_left = round(self.specifications.warm_up[0] / self.specifications.duration * num_frames)
            weight[:, :warm_up_left] = 0.0
            warm_up_right = round(self.specifications.warm_up[1] / self.specifications.duration * num_frames)
            weight[:, num_frames - warm_up_right :] = 0.0

            if self.specifications.powerset:
                multilabel = self.pyan_nn.powerset.to_multilabel(prediction)
                permutated_target, _ = permutate(multilabel, labels)

                permutated_target_powerset = self.pyan_nn.powerset.to_powerset(permutated_target.float())
                loss = self.segmentation_loss(prediction, permutated_target_powerset, weight=weight)

            else:
                permutated_prediction, _ = permutate(labels, prediction)
                loss = self.segmentation_loss(permutated_prediction, labels, weight=weight)

            return {"loss": loss, "logits": prediction}

        return {"logits": prediction}

    def setup_loss_func(self):
        """setup the loss function is self.specifications.powerset is True."""
        if self.specifications.powerset:
            self.pyan_nn.powerset = Powerset(
                len(self.specifications.classes),
                self.specifications.powerset_max_classes,
            )

    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Permutation-invariant segmentation loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes) torch.Tensor
            Permutated speaker activity predictions.
        target : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        weight : (batch_size, num_frames, 1) torch.Tensor, optional
            Frames weight.

        Returns
        -------
        seg_loss : torch.Tensor
            Permutation-invariant segmentation loss
        """

        if self.specifications.powerset:
            # `clamp_min` is needed to set non-speech weight to 1.
            class_weight = torch.clamp_min(self.pyan_nn.powerset.cardinality, 1.0) if self.weigh_by_cardinality else None
            seg_loss = nll_loss(
                permutated_prediction,
                torch.argmax(target, dim=-1),
                class_weight=class_weight,
                weight=weight,
            )
        else:
            seg_loss = binary_cross_entropy(permutated_prediction, target.float(), weight=weight)

        return seg_loss


    @classmethod
    def from_pyannote_model(cls, pretrained, with_weight=True, sincnet=None):
        """Copy the weights and architecture of a pre-trained Pyannote model.

        Args:
            pretrained (pyannote.core.Model): pretrained pyannote segmentation model.
        """
        # Initialize model:
        specifications = copy.deepcopy(pretrained.specifications)

        # Copy pretrained model hyperparameters:
        chunk_duration = specifications.duration
        max_speakers_per_frame = specifications.powerset_max_classes
        weigh_by_cardinality = False
        min_duration = specifications.min_duration
        warm_up = specifications.warm_up
        max_speakers_per_chunk = len(specifications.classes)

        config = SegmentationModelConfig(
            chunk_duration=chunk_duration,
            max_speakers_per_frame=max_speakers_per_frame,
            weigh_by_cardinality=weigh_by_cardinality,
            min_duration=min_duration,
            warm_up=warm_up,
            max_speakers_per_chunk=max_speakers_per_chunk,
            sincnet = sincnet,
        )

        model:SegmentationModel = cls(config)

        # Copy pretrained model weights:
        if with_weight:
            print('before copy:', model.pyan_nn._sincnet_pool)
            model.pyan_nn.hparams = copy.deepcopy(pretrained.hparams)
            # 구조적으로 다른 SincNet과 SincNetPool 간의 가중치 복사
            # model.pyan_nn._sincnet_pool = copy.deepcopy(pretrained.sincnet)
            # model.pyan_nn._sincnet_pool.load_state_dict(pretrained.sincnet.state_dict())
            model.pyan_nn._sincnet_pool.wav_norm1d = copy.deepcopy(pretrained.sincnet.wav_norm1d)
            model.pyan_nn._sincnet_pool.wav_norm1d.load_state_dict(pretrained.sincnet.wav_norm1d.state_dict())
            model.pyan_nn._sincnet_pool.conv1d = copy.deepcopy(pretrained.sincnet.conv1d)
            model.pyan_nn._sincnet_pool.conv1d.load_state_dict(pretrained.sincnet.conv1d.state_dict())
            model.pyan_nn._sincnet_pool.pool1d = copy.deepcopy(pretrained.sincnet.pool1d)
            model.pyan_nn._sincnet_pool.pool1d.load_state_dict(pretrained.sincnet.pool1d.state_dict())
            model.pyan_nn._sincnet_pool.norm1d = copy.deepcopy(pretrained.sincnet.norm1d)
            model.pyan_nn._sincnet_pool.norm1d.load_state_dict(pretrained.sincnet.norm1d.state_dict())
            # no model.pyan_nn._sincnet_pool.post_pool layer in SincNet
            model.pyan_nn.lstm = copy.deepcopy(pretrained.lstm)
            model.pyan_nn.lstm.load_state_dict(pretrained.lstm.state_dict())
            model.pyan_nn.linear = copy.deepcopy(pretrained.linear)
            model.pyan_nn.linear.load_state_dict(pretrained.linear.state_dict())
            model.pyan_nn.classifier = copy.deepcopy(pretrained.classifier)
            model.pyan_nn.classifier.load_state_dict(pretrained.classifier.state_dict())
            model.pyan_nn.activation = copy.deepcopy(pretrained.activation)
            model.pyan_nn.activation.load_state_dict(pretrained.activation.state_dict())

        return model

    def to_pyannote_model(self):
        """Convert the current model to a pyannote segmentation model for use in pyannote pipelines."""

        seg_model = PyanNet(sincnet={"stride": 10})
        seg_model.hparams.update(self.pyan_nn.hparams)

        seg_model._sincnet = copy.deepcopy(self.pyan_nn._sincnet_pool)
        seg_model._sincnet.load_state_dict(self.pyan_nn._sincnet_pool.state_dict())

        # if self.model.post_pool:
        #     seg_model.post_pool = copy.deepcopy(self.model.post_pool)
        #     seg_model.post_pool.load_state_dict(self.model.post_pool.state_dict())
        #     print('post_pool copied')

        seg_model.lstm = copy.deepcopy(self.pyan_nn.lstm)
        seg_model.lstm.load_state_dict(self.pyan_nn.lstm.state_dict())

        seg_model.linear = copy.deepcopy(self.pyan_nn.linear)
        seg_model.linear.load_state_dict(self.pyan_nn.linear.state_dict())

        seg_model.classifier = copy.deepcopy(self.pyan_nn.classifier)
        seg_model.classifier.load_state_dict(self.pyan_nn.classifier.state_dict())

        seg_model.activation = copy.deepcopy(self.pyan_nn.activation)
        seg_model.activation.load_state_dict(self.pyan_nn.activation.state_dict())

        seg_model.specifications = self.specifications

        return seg_model

    @classmethod
    def from_checkpoint(cls, checkpoint: str|Path) -> 'SegmentationModel':
        """Load a model from a checkpoint directory.

        Args:
            checkpoint (str): Path to the checkpoint directory containing `config.json` and `model.safetensors`.

        Returns:
            SegmentationModel: An instance of SegmentationModel with loaded configuration and weights.
        """
        import json
        from safetensors.torch import load_file
        # Load configuration
        config_path = Path(checkpoint) / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = SegmentationModelConfig(**config_dict)

        # Initialize model with configuration
        model = cls(config)

        # Load model weights
        model_weights_path = Path(checkpoint) / "model.safetensors"
        state_dict = load_file(model_weights_path)
        model.load_state_dict(state_dict)

        return model

    @cached_property
    def receptive_field(self) -> SlidingWindow:
        start, size, step = self.pyan_nn._sincnet_pool.receptive_field()
        sr = 16000
        return SlidingWindow(
            start=start / sr,
            duration= size / sr,
            step=step / sr,
        )

if __name__ == "__main__":
    ...