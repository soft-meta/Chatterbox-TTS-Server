import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the repository tests independent from the external softmeta-chatterbox-v2
# package that is installed by the Colab notebook.
if 'engine' not in sys.modules:
    fake = types.ModuleType('engine')
    class EngineService:
        def __init__(self, device='auto'):
            self.device = device
            self.loaded_model = None
        def status(self):
            return {
                'device': self.device, 'model_name': self.loaded_model,
                'loading': False, 'error': None, 'sample_rate': 24000,
            }
        def load(self, model_name):
            self.loaded_model = model_name
        def unload(self):
            self.loaded_model = None
    fake.EngineService = EngineService
    sys.modules['engine'] = fake
