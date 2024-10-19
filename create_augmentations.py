import re
import warnings
import torch
from torch import nn
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from random import choices
from tqdm import tqdm
import librosa


class RMSE(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


possible_descriptions = [
    "happy rock",
    "energetic EDM",
    "sad jazz",
    "punk rock",
    "upbeat instrumental",
    "video game theme",
    "children's song",
    "indian dance music",
    "k-pop",
    "electro-hop",
    "pop rap",
    "inspirational love song",
    "disco song",
    "acoustic guitar chords",
    "funy piano chords",
    "pop music"
    "energetic rock",
    "slow rock",
    "love ballad",
    "stereotypical pop muisc",
    "techno",
    "rave music",
    "strings",
]


if __name__ == "__main__":
  model = MusicGen.get_pretrained("facebook/musicgen-melody")
  n_samples_per_acc: int = 3
  mirst500_dir: Path = Path("your/path/to/mirst500")
  files = tuple(mirst500_dir.rglob("vocals.wav"))  # NOTE: You need to have vocals and accompaniments for each song in MIRST500 using some source separation model
  
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for vocals_file in (pbar := tqdm(files)):
        pbar.set_description(f"processing file {str(vocals_file)}")
        try:
            this_song_possible_descriptions = set(possible_descriptions)  # by default this song can have any description
            AI_vocals_present = []
            for f in vocals_file.parent.rglob("AI_accompaniment_*wav"):
                match = re.match(r".*?AI_accompaniment_(.+?)\.wav", str(f)).group(1).replace("_", " ")
                this_song_possible_descriptions.discard(match)  # If there is a generated ai song for such prompt, we dont generate it again

            descriptions = choices(tuple(this_song_possible_descriptions), k=n_samples_per_acc)
            accompaniment_file = vocals_file.parent / "accompaniment.wav"
            if not accompaniment_file.exists():
                print("Can not find accompaniment for track ", str(accompaniment_file), " skipping")
                continue

            accompaniment_signal, sr = torchaudio.load(accompaniment_file)
            vocals_signal, _         = torchaudio.load(vocals_file)


            output_wavs = torch.zeros_like(accompaniment_signal).repeat(n_samples_per_acc, 1).unsqueeze(1)
            model.set_generation_params(duration=vocals_signal.shape[-1] / sr)  # set max generation length

            created_chunks = model.generate_with_chroma(descriptions, accompaniment_signal[None].expand(n_samples_per_acc, -1, -1), sr, progress=False)
            created_chunks_resampled = torchaudio.functional.resample(created_chunks, model.sample_rate, sr)
            created_chunks_resampled = created_chunks_resampled[..., :vocals_signal.shape[-1]]

            for idx, one_wav in enumerate(created_chunks_resampled):
                output_file = vocals_file.parent / ("AI_accompaniment_" + descriptions[idx].replace(" ", "_"))
                audio_write(str(output_file), one_wav.cpu(), sr, strategy="peak", loudness_compressor=False)
        except Exception as e:
            print("Exception ", str(e), " while processing file ", str(vocals_file))
        else:
            print("Generated outputs for file ", str(vocals_file))
