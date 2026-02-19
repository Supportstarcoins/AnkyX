from llm_providers import OpenAIProvider


class DummyOpenAI(OpenAIProvider):
    def __init__(self):
        self.calls = 0

    def chat(self, messages, settings):
        self.calls += 1
        if self.calls == 1:
            return "not json"
        return '{"cards":[{"type":"qa","front":"Q","back":"A","tags":[],"source":{"kind":"text","ref":"r"},"media":{"type":"null","asset_id":null}}]}'


def test_json_retry_success():
    provider = DummyOpenAI()
    payload = provider.generate_json("card_set_qa.json", "input", {})
    assert payload["cards"][0]["type"] == "qa"
    assert provider.calls == 2
