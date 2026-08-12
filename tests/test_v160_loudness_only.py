from pathlib import Path
import numpy as np
import soundfile as sf
from queue_manager import QueueManager


def test_loudness_only_postprocessing_produces_valid_loud_wav(tmp_path):
    sr=24000
    t=np.arange(sr*3)/sr
    audio=(0.015*np.sin(2*np.pi*180*t)).astype('float32')
    path=tmp_path/'in.wav'
    sf.write(path,audio,sr)
    profile=QueueManager._normalise_loudness(path)
    info=sf.info(path)
    data,_=sf.read(path,dtype='float32')
    peak=float(np.max(np.abs(data)))
    rms=float(np.sqrt(np.mean(np.square(data))+1e-12))
    assert info.samplerate==sr and info.channels==1 and info.duration>2.9
    assert profile['profile']=='loudness-only'
    assert profile['target_lufs']==-12.5
    assert peak <= 1.0
    assert rms > 0.05  # clearly louder than the -36 dBFS test source
