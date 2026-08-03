import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from avatar_engine import AvatarEngineService
from models import VideoJobCreate
from scripts.patch_longcat_runtime import patch_runtime


ROOT = Path(__file__).resolve().parents[1]


class LongCatAvatarRegressionTests(unittest.TestCase):
    def test_model_readiness_requires_every_indexed_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = root / "quantized_model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}))
            (root / "part-1.safetensors").touch()
            self.assertFalse(AvatarEngineService._indexed_model_complete(index))
            (root / "part-2.safetensors").touch()
            self.assertTrue(AvatarEngineService._indexed_model_complete(index))

    def test_avatar_process_is_headless_and_single_gpu_safe(self):
        with patch.dict(os.environ, {"MPLBACKEND": "module://matplotlib_inline.backend_inline"}):
            env = AvatarEngineService._headless_process_env()
        self.assertEqual(env["MPLBACKEND"], "Agg")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(env["NCCL_P2P_DISABLE"], "1")

    def test_longcat_duration_math(self):
        self.assertEqual(AvatarEngineService._longcat_segment_count(3.0), 1)
        self.assertEqual(AvatarEngineService._longcat_segment_count(3.72), 1)
        self.assertEqual(AvatarEngineService._longcat_segment_count(3.73), 2)
        self.assertEqual(AvatarEngineService._longcat_segment_count(6.92), 2)

    def test_request_defaults_to_realistic_longcat(self):
        request = VideoJobCreate(
            avatar_filename="portrait.png",
            audio_job_id="audio-id",
            consent=True,
        )
        self.assertEqual(request.engine, "auto")
        self.assertEqual(request.render_mode, "continuous")
        self.assertEqual(request.segment_seconds, 300)
        self.assertEqual(request.native_resolution, "720p")
        self.assertEqual(request.motion_style, "natural")

    def test_installer_is_pinned_selective_and_safe(self):
        installer = (ROOT / "scripts/install_longcat_avatar_a100.sh").read_text()
        self.assertIn("6b3f4b8582a8bc3f20f795735f5383716c4ba794", installer)
        self.assertIn("92016c71d5d318d0f5d84e4db30015a571484ab6", installer)
        self.assertIn('"base_model_int8/**"', installer)
        self.assertIn("safe_content_child", installer)
        self.assertIn("LongCat Video Avatar 1.5 is ready", installer)
        self.assertIn("libsndfile1|tritonserverclient", installer)
        self.assertIn(".softmeta_longcat_avatar_v15_v105", installer)
        self.assertNotIn('"lora/dmd_lora.safetensors"', installer)
        self.assertIn("model_shards_ok", installer)
        self.assertIn('"accelerate>=1.2,<2"', installer)

    def test_runtime_patch_is_guarded(self):
        source = (ROOT / "scripts/patch_longcat_runtime.py").read_text()
        self.assertIn("refusing an unsafe patch", source)
        self.assertIn("SOFTMETA_LONGCAT_SEED", source)
        self.assertIn("segment_idx == num_segments - 1", source)

    def test_runtime_patch_applies_to_pinned_upstream_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            pipeline_dir = repo / "longcat_video"
            pipeline_dir.mkdir()
            entrypoint = repo / "run_demo_avatar_single_audio_to_video.py"
            entrypoint.write_text(
                "import os\nimport argparse\nimport torch\nimport numpy as np\n"
                "def torch_gc():\n"
                "    torch.cuda.empty_cache()\n"
                "    torch.cuda.ipc_collect()\n"
                "def generate():\n"
                "    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder=\"text_encoder\", torch_dtype=torch.bfloat16)\n"
                "    dit = load_quantized_dit(checkpoint_dir)\n"
                "    # initialize audio models\n"
                "    global_seed = 42\n"
                "    all_generated_frames = video\n"
                "    for segment_idx in range(1, num_segments):\n"
                "        all_generated_frames.extend(new_video[num_cond_frames:])\n"
                "        if cp_rank == 0:\n"
                "            output_tensor = torch.from_numpy(np.array(all_generated_frames))\n"
                "            save_video_ffmpeg(output_tensor, os.path.join(output_dir, f\"video_continue_{segment_idx+1}\"), raw_speech_path, fps=save_fps, quality=5)\n"
                "            del output_tensor\n"
            )
            pipeline = pipeline_dir / "pipeline_longcat_video_avatar.py"
            pipeline.write_text(
                "import os\n"
                "        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state\n"
                "        if self.text_encoder is not None:\n"
                "            self.text_encoder = self.text_encoder.to(device, non_blocking=True)\n"
            )
            patch_runtime(repo)
            patched_entrypoint = entrypoint.read_text()
            patched_pipeline = pipeline.read_text()
            self.assertIn("SOFTMETA_LONGCAT_SEED", patched_entrypoint)
            self.assertLess(
                patched_entrypoint.index("load_quantized_dit"),
                patched_entrypoint.index("SoftMeta: load the large T5 encoder"),
            )
            self.assertIn("segment_idx == num_segments - 1", patched_entrypoint)
            self.assertNotIn("all_generated_frames", patched_entrypoint)
            self.assertIn("softmeta_concat_video_chunks", patched_entrypoint)
            self.assertIn("encoder_device", patched_pipeline)
            self.assertIn("SOFTMETA_LONGCAT_TEXT_ENCODER_CPU", patched_pipeline)
            ast.parse(patched_entrypoint)
            patch_runtime(repo)

    def test_colab_uses_longcat_only_when_enabled(self):
        notebook_path = ROOT / "colab/SoftMeta_Chatterbox_TTS_Colab_v1.0.5.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("INSTALL_AVATAR_RUNTIME = False", source)
        self.assertIn("install_longcat_avatar_a100.sh", source)
        self.assertIn('env_python("longcat310")', source)
        self.assertIn('"SOFTMETA_LONGCAT_MODELS": "/content/longcat_models"', source)
        self.assertNotIn("install_ditto", source.lower())


if __name__ == "__main__":
    unittest.main()
