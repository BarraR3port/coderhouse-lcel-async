from collections.abc import Iterator

import pytest

from tests.fake_openai import FakeOpenAI, running_fake_openai

OPENAI_VARS = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_BASE")
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*OPENAI_VARS, *PROXY_VARS, *(proxy.lower() for proxy in PROXY_VARS)):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


@pytest.fixture
def fake_openai(api_key: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOpenAI]:
    with running_fake_openai() as fake:
        monkeypatch.setenv("OPENAI_BASE_URL", fake.base_url)
        yield fake
