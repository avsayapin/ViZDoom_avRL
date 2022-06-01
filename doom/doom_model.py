import torch
from torch import nn
import torchaudio
import torch.nn.functional as F

from sample_factory.algorithms.appo.model_utils import get_obs_shape, nonlinearity, create_standard_encoder, EncoderBase, \
    register_custom_encoder
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements
from sample_factory.utils.utils import log


class VizdoomEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        self.basic_encoder = create_standard_encoder(cfg, obs_space, timing)
        self.encoder_out_size = self.basic_encoder.encoder_out_size
        obs_shape = get_obs_shape(obs_space)

        self.measurements_head = None
        if 'measurements' in obs_shape:
            self.measurements_head = nn.Sequential(
                nn.Linear(obs_shape.measurements[0], 128),
                nonlinearity(cfg),
                nn.Linear(128, 128),
                nonlinearity(cfg),
            )
            measurements_out_size = calc_num_elements(self.measurements_head, obs_shape.measurements)
            self.encoder_out_size += measurements_out_size

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict)

        if self.measurements_head is not None:
            measurements = self.measurements_head(obs_dict['measurements'].float())
            x = torch.cat((x, measurements), dim=1)

        return x

class VizdoomSoundEncoder(EncoderBase):
    """
    audio_encoder_type: Type of audio encoder to use (available: "logmel", "fft", "samples")
    """
    def __init__(self, cfg, obs_space, timing, audio_encoder_type="logmel"):
        super().__init__(cfg, timing)
        self.audio_encoder_type = audio_encoder_type
        # self.sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_rate = cfg.sampling_rate
        # self.frameskip = DEFAULT_FRAMESKIP
        self.frameskip = cfg.num_frames

        self.basic_encoder = create_standard_encoder(cfg, obs_space, timing)
        self.encoder_out_size = self.basic_encoder.encoder_out_size
        obs_shape = get_obs_shape(obs_space)

        self.sound_head = None
        if 'sound' in obs_shape:
            if self.audio_encoder_type == "fft":
                self.sound_head = nn.Sequential(
                    SimpleFFTAudioEncoder(self.sample_rate, self.frameskip),
                    nn.Flatten(),
                    nn.ReLU()
                )
            elif self.audio_encoder_type == "logmel":
                self.sound_head = nn.Sequential(
                    LogMelAudioEncoder(self.sample_rate, self.frameskip),
                    nn.Flatten(),
                    nn.ReLU(),
                )
            elif self.audio_encoder_type == "samples":
                self.sound_head = nn.Sequential(
                    SimpleRawSamplesEncoder(self.sample_rate, self.frameskip),
                    nn.Flatten(),
                    nn.ReLU(),
                )
            else:
                raise NotImplementedError("Audio encoder {} not implemented".format(self.audio_encoder_type))
            sound_out_size = calc_num_elements(self.sound_head, obs_shape.sound)
            self.encoder_out_size += sound_out_size

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict)


        if self.sound_head is not None:

            # Normalize to [-1, 1] (default for audio)
            obs_dict['sound'].mul_(1.0 / 32767)

            sound = self.sound_head(obs_dict['sound'].float())

            x = torch.cat((x, sound), dim=1)

        return x

class SimpleFFTAudioEncoder(nn.Module):
    """Very simple audio processing:
    FFT -> magnitude -> log -> subsample (maxpool) -> few linear layers
    """
    def __init__(self, sample_rate, frameskip):
        super(SimpleFFTAudioEncoder, self).__init__()
        self.num_to_subsample = 8
        # ViZDoom runs at 35 fps, but we will get frameskip number of
        # frames in total (concatenated)
        self.num_samples = (sample_rate / 35) * frameskip
        self.num_frequencies = self.num_samples / 2
        assert int(self.num_samples) == self.num_samples
        self.num_samples = int(self.num_samples)
        self.num_frequencies = int(self.num_frequencies)

        self.hamming_window = torch.hamming_window(self.num_samples)

        # Subsampler
        self.pool = torch.nn.MaxPool1d(self.num_to_subsample)

        # Encoder (small MLP)
        self.linear1 = torch.nn.Linear(int(self.num_frequencies / self.num_to_subsample), 256)
        self.linear2 = torch.nn.Linear(256, 256)

    def _torch_1d_fft_magnitude(self, x):
        """Perform 1D FFT on x with shape (batch_size, num_samples), and return magnitudes"""
        # Apply hamming window
        if x.device != self.hamming_window.device:
            self.hamming_window = self.hamming_window.to(x.device)
        x = x * self.hamming_window
        # Add zero imaginery parts
        x = torch.stack((x, torch.zeros_like(x)), dim=-1)
        ffts = torch.fft(x, signal_ndim=1)
        # Remove mirrored part
        ffts = ffts[:, :(ffts.shape[1] // 2), :]
        # To magnitudes
        mags = torch.sqrt(ffts[..., 0]**2 + ffts[..., 1]**2)
        return mags

    def _encode_channel(self, x):
        """Shape of x: [batch_size, num_samples]"""
        mags = self._torch_1d_fft_magnitude(x)
        mags = torch.log(mags + 1e-5)

        # Add and remove "channel" dim...
        x = self.pool(mags[:, None, :])[:, 0, :]
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        return x

    def forward(self, x):
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]

        x1 = self._encode_channel(x1)
        x2 = self._encode_channel(x2)
        x = torch.cat((x1, x2), dim=1)
        return x


class VizdoomSoundEncoderFFT(VizdoomSoundEncoder):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, obs_space, timing, audio_encoder_type="fft")

def register_models():
    register_custom_encoder('vizdoom', VizdoomEncoder)
    register_custom_encoder('vizdoomSound', VizdoomSoundEncoder)
    register_custom_encoder('vizdoomSoundFFT', VizdoomSoundEncoderFFT)
