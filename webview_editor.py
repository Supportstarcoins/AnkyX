import importlib
import importlib.util
import json
import threading
from typing import Any, Dict

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
    def __init__(self, title: str = "Редактор конспекта") -> None:
        self.title = title
        self.bridge = QuillEditorBridge()
        self._window = None
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()

    def is_running(self) -> bool:
        return self._window is not None

    def show(self) -> None:
        if not QUILL_WEBVIEW_AVAILABLE:
            raise RuntimeError("pywebview не установлен. Установите: pip install pywebview")
        if self._thread and self._thread.is_alive():
            return
        webview = importlib.import_module("webview")
        html = self._build_html()
        self._window = webview.create_window(
            self.title,
            html=html,
            width=980,
            height=720,
            js_api=self.bridge,
        )

        def _start():
            webview.start(self._on_ready, debug=False, gui="tkinter")

        self._thread = threading.Thread(target=_start, daemon=True)
        self._thread.start()

    def _on_ready(self) -> None:
        self._ready_event.set()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready_event.wait(timeout)

    def set_content(self, payload: Dict[str, Any]) -> None:
        if not self._window:
            return
        self.wait_ready()
        data = json.dumps(payload, ensure_ascii=False)
        self._window.evaluate_js(f"setEditorContent({data});")

    def get_content(self) -> Dict[str, Any]:
        if not self._window:
            return {}
        self.wait_ready()
        result = self._window.evaluate_js("JSON.stringify(getEditorContent())")
        try:
            return json.loads(result) if result else {}
        except json.JSONDecodeError:
            return {}

    def get_selection(self) -> Dict[str, Any]:
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
