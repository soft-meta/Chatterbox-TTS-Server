from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generation_code_never_requests_colab_disconnect():
    queue = (ROOT / 'queue_manager.py').read_text(encoding='utf-8')
    server = (ROOT / 'server.py').read_text(encoding='utf-8')
    assert 'unassign' not in queue.lower()
    assert 'runtime/disconnect' not in queue
    # The server may expose the explicit runtime endpoint, but no completed-job
    # handler should call that endpoint/helper automatically.
    assert server.count('@app.post("/api/runtime/disconnect")') == 1
    assert '_request_colab_unassign' in server
    assert 'completed_at' not in server[server.index('async def disconnect_colab_runtime'):server.index('@app.post("/tts")')]


def test_colab_start_cell_supervises_server_instead_of_auto_stopping():
    nb = (ROOT / 'colab' / 'SoftMeta_Chatterbox_TTS_Colab_v1.6.0.ipynb').read_text(encoding='utf-8')
    assert 'KEEP_RUNTIME_ACTIVE' in nb
    assert 'Server exited unexpectedly with code' in nb
    assert 'restarting...' in nb
    assert 'SOFTMETA_ASR_MODEL' not in nb
    assert 'SOFTMETA_SPEAKER_CACHE' not in nb
