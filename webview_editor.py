import importlib
import importlib.util
import json
import threading
from typing import Any, Dict, Optional

QUILL_WEBVIEW_AVAILABLE = importlib.util.find_spec("webview") is not None


class QuillEditorBridge:
    def __init__(self) -> None:
        self.last_content: Dict[str, Any] | None = None
        self.last_selection: Dict[str, Any] | None = None

    def setContent(self, html_or_delta: Dict[str, Any]) -> bool:
        self.last_content = html_or_delta
        return True

    def getContent(self) -> Dict[str, Any] | None:
        return self.last_content

    def getSelectionHtmlOrDelta(self, payload: Dict[str, Any]) -> bool:
        self.last_selection = payload
        return True


class QuillEditorWindow:
    def __init__(self, title: str = "Редактор конспекта", on_close: Optional[callable] = None) -> None:
        self.title = title
        self.bridge = QuillEditorBridge()
        self._window = None
        self._ready_event = threading.Event()
        self._closed = True
        self._on_close_callback = on_close

    def is_running(self) -> bool:
        return self._window is not None and not self._closed

    def show(self) -> None:
        if not QUILL_WEBVIEW_AVAILABLE:
            raise RuntimeError("pywebview не установлен. Установите: pip install pywebview")
        webview = importlib.import_module("webview")
        if self._window is not None and not self._closed:
            self._bring_to_front()
            return
        html = self._build_html()
        self._ready_event.clear()
        self._window = webview.create_window(
            "Редактор конспекта (Quill)",
            html=html,
            width=1100,
            height=700,
            resizable=True,
            js_api=self.bridge,
        )
        self._closed = False
        try:
            self._window.events.loaded += self._on_ready
            self._window.events.closed += self._on_closed
        except Exception:
            pass
        self._bring_to_front()

    def _bring_to_front(self) -> None:
        if not self._window:
            return
        try:
            self._window.bring_to_front()
        except Exception:
            pass

    def _on_ready(self) -> None:
        self._ready_event.set()

    def _on_closed(self) -> None:
        self._closed = True
        self._window = None
        self._ready_event.clear()
        if self._on_close_callback:
            self._on_close_callback()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready_event.wait(timeout)

    def set_html(self, html: str) -> None:
        if not self._window:
            return
        self.wait_ready()
        data = json.dumps({"html": html}, ensure_ascii=False)
        self._window.evaluate_js(f"setEditorContent({data});")

    def get_html(self) -> str:
        payload = self._get_content_payload()
        return payload.get("html", "")

    def get_delta(self) -> Dict[str, Any]:
        payload = self._get_content_payload()
        delta = payload.get("delta")
        return delta if isinstance(delta, dict) else {}

    def get_selection_html(self) -> str:
        payload = self._get_selection_payload()
        return payload.get("html", "")

    def get_selection_delta(self) -> Dict[str, Any]:
        payload = self._get_selection_payload()
        delta = payload.get("delta")
        return delta if isinstance(delta, dict) else {}

    def _get_content_payload(self) -> Dict[str, Any]:
        if not self._window:
            return {}
        self.wait_ready()
        result = self._window.evaluate_js("JSON.stringify(getEditorContent())")
        try:
            return json.loads(result) if result else {}
        except json.JSONDecodeError:
            return {}

    def _get_selection_payload(self) -> Dict[str, Any]:
        if not self._window:
            return {}
        self.wait_ready()
        result = self._window.evaluate_js("JSON.stringify(getSelectionPayload())")
        try:
            return json.loads(result) if result else {}
        except json.JSONDecodeError:
            return {}

    def _build_html(self) -> str:
        return """
<!doctype html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <link href=\"https://cdn.jsdelivr.net/npm/quill@1.3.7/dist/quill.snow.css\" rel=\"stylesheet\">
  <style>
    body { margin: 0; font-family: 'Segoe UI', sans-serif; background: #0B0D12; color: #E8ECF4; }
    #toolbar { border: none; background: #111522; }
    #toolbar .ql-picker { color: #E8ECF4; }
    #editor { height: calc(100vh - 120px); background: #0F1422; color: #E8ECF4; }
    .actions { display: flex; gap: 8px; padding: 8px 12px; background: #111522; }
    .actions button { background: #1F2937; color: #E8ECF4; border: none; padding: 8px 12px; cursor: pointer; border-radius: 6px; }
  </style>
</head>
<body>
  <div id=\"toolbar\">
    <span class=\"ql-formats\">
      <button class=\"ql-bold\"></button>
      <button class=\"ql-underline\"></button>
    </span>
    <span class=\"ql-formats\">
      <select class=\"ql-color\"></select>
      <select class=\"ql-background\"></select>
    </span>
    <span class=\"ql-formats\">
      <button class=\"ql-list\" value=\"ordered\"></button>
      <button class=\"ql-list\" value=\"bullet\"></button>
    </span>
    <span class=\"ql-formats\">
      <button id=\"insert-table\" type=\"button\">▦</button>
    </span>
  </div>
  <div id=\"editor\"></div>
  <div class=\"actions\">
    <button id=\"send-content\" type=\"button\">Отправить контент в приложение</button>
    <button id=\"send-selection\" type=\"button\">Отправить выделение</button>
  </div>
  <script src=\"https://cdn.jsdelivr.net/npm/quill@1.3.7/dist/quill.min.js\"></script>
  <script>
    const quill = new Quill('#editor', {
      theme: 'snow',
      modules: { toolbar: '#toolbar' }
    });

    function deltaToHtml(delta) {
      const temp = document.createElement('div');
      const tmpQuill = new Quill(temp);
      tmpQuill.setContents(delta);
      return tmpQuill.root.innerHTML;
    }

    window.getEditorContent = function() {
      return { html: quill.root.innerHTML, delta: quill.getContents() };
    }

    window.getSelectionPayload = function() {
      const range = quill.getSelection();
      if (!range || range.length === 0) {
        return window.getEditorContent();
      }
      const delta = quill.getContents(range.index, range.length);
      let html = '';
      if (typeof quill.getSemanticHTML === 'function') {
        html = quill.getSemanticHTML(range.index, range.length);
      } else {
        html = deltaToHtml(delta);
      }
      return { html: html, delta: delta };
    }

    window.setEditorContent = function(payload) {
      if (!payload) return;
      if (typeof payload === 'string') {
        quill.clipboard.dangerouslyPasteHTML(payload);
      } else if (payload.delta) {
        quill.setContents(payload.delta);
      } else if (payload.html) {
        quill.clipboard.dangerouslyPasteHTML(payload.html);
      }
    }

    document.getElementById('insert-table').addEventListener('click', () => {
      const tableHtml = '<table style="width:100%; border-collapse: collapse;" border="1">' +
        '<tr><td> </td><td> </td></tr>' +
        '<tr><td> </td><td> </td></tr>' +
        '</table><p><br></p>';
      const range = quill.getSelection(true);
      quill.clipboard.dangerouslyPasteHTML(range ? range.index : 0, tableHtml);
    });

    document.getElementById('send-content').addEventListener('click', () => {
      if (window.pywebview) {
        window.pywebview.api.setContent(window.getEditorContent());
      }
    });

    document.getElementById('send-selection').addEventListener('click', () => {
      if (window.pywebview) {
        window.pywebview.api.getSelectionHtmlOrDelta(window.getSelectionPayload());
      }
    });
  </script>
</body>
</html>
"""
