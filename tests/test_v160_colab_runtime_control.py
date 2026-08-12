import os, sys, types, importlib
from pathlib import Path


def _load_server(monkeypatch, tmp_path):
    # Import against the real working tree; the test only calls the isolated helper.
    for name in ['server']:
        sys.modules.pop(name, None)
    import server
    return server


def test_disconnect_helper_posts_to_colab_unassign(monkeypatch, tmp_path):
    server=_load_server(monkeypatch,tmp_path)
    monkeypatch.setenv('TBE_RUNTIME_ADDR','127.0.0.1:9999')
    seen={}
    class Resp:
        status=200
        def __enter__(self): return self
        def __exit__(self,*a): return False
    def fake_urlopen(req, timeout=0):
        seen['url']=req.full_url; seen['method']=req.get_method(); seen['timeout']=timeout
        return Resp()
    monkeypatch.setattr(server.urllib.request,'urlopen',fake_urlopen)
    server._request_colab_unassign()
    assert seen['url']=='http://127.0.0.1:9999/unassign'
    assert seen['method']=='POST'


def test_ui_has_manual_disconnect_but_no_auto_disconnect_hook():
    root=Path(__file__).resolve().parents[1]
    html=(root/'ui/index.html').read_text()
    js=(root/'ui/app.js').read_text()
    assert 'id="disconnect-colab"' in html
    assert '/api/runtime/disconnect' in js
    # Only explicit click handler should invoke it; no completed-job trigger string around it.
    assert js.count('/api/runtime/disconnect') == 1
