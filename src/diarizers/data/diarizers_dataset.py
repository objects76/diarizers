from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from dataclasses import dataclass
from pyannote.core import Segment
from pathlib import Path
import numpy as np
from datasets import load_dataset
import ffmpeg

@dataclass
class DiarizersAnnotation:
    pcm1d_norm: np.ndarray
    segments: list[Segment]
    speakers: list[str]

    def __init__(self, data:dict) -> None:
        self.pcm1d_norm= data['audio']['array']
        self.sr=data['audio']['sampling_rate']
        self.timestamps_start = data['timestamps_start']
        self.timestamps_end= data['timestamps_end']
        # self.speakers= data['speakers']
        # remap long speaker name to 1,2,3
        unique_speakers = {name: f"SPEAKER_{idx:02}"
                           for idx, name in enumerate(set(data['speakers']), start=1)}
        self.speakers = [unique_speakers[name] for name in data['speakers']]

        self.segments = [Segment(s,e) for s,e in zip(self.timestamps_start, self.timestamps_end)]
        assert self.sr == 16000
        assert len(self.segments) == len(self.speakers)

    def __repr__(self):
        return (
            f"AudioAnnotation( waveform={self.pcm1d_norm.shape[0]/self.sr:.1f}sec, "
            f"n_seg={len(self.speakers)}, {set(self.speakers)} )"
        )

    @property
    def annotation(self):
        anno = Annotation(uri="ds")
        for seg, label in zip(self.segments, self.speakers):
            anno[seg] = label
        return anno

    def der(self, hypothesis:Annotation, metric:DiarizationErrorRate|None = None):
        metric = metric or DiarizationErrorRate()
        # return metric(self.annotation, hypothesis, detailed=True, uem=None)
        metric(self.annotation, hypothesis, detailed=True)
        df = metric.report()
        if len(df) == 2:
            df = df.drop(df.index[0])
        df.columns = [f'({col[1]})' if col[1] else col[0] for col in df.columns]
        return df.to_string(index=False, float_format="{:.1f}".format)  # Removes both[1]

    def save(self, audio_path:str|Path):
        pcm_data = self.pcm1d_norm.astype(np.float32)
        (
            ffmpeg
            .input('pipe:0', format='f32le', ar=self.sr, ac=1)
            .output(str(audio_path), acodec='aac')
            .global_args('-loglevel', 'warning', '-y')
            .run(input=pcm_data.tobytes())
        )

        anno = self.annotation
        rttm_path = Path(audio_path).with_suffix('.rttm')
        with open(rttm_path, "w") as fp:
            anno.uri = rttm_path.stem
            anno.write_rttm(fp)

    @staticmethod
    def from_dataset(ds):
        return [DiarizersAnnotation(d) for d in ds]


    @staticmethod
    def from_huggingface(path, subset = None, *, split = None):
        ds = load_dataset(path, name=subset, split=split)
        return [DiarizersAnnotation(d) for d in ds]


def dump(ds:list[DiarizersAnnotation], outdir:Path|str, prefix:str=''):
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    for idx, sample in enumerate(ds):
        audio_path:Path = outdir / f'{prefix}{idx:04d}.m4a'
        sample.save(audio_path)
