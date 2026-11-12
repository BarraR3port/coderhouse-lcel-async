from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

import main as app
from tests.fake_openai import FakeOpenAI

ANSWER = "LCEL compone prompt, modelo y parser con el operador |."


class AsyncOnlyModel(GenericFakeChatModel):
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("la cadena se ejecutó con invoke() en lugar de ainvoke()")

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return GenericFakeChatModel._generate(self, *args, **kwargs)


def async_only_model(answer: str) -> AsyncOnlyModel:
    return AsyncOnlyModel(messages=iter([AIMessage(content=answer)]))


async def test_chain_returns_plain_text_not_a_message() -> None:
    chain = app.build_chain(async_only_model(ANSWER))

    assert await chain.ainvoke({"pregunta": "¿Qué es LCEL?"}) == ANSWER


def test_chain_is_prompt_then_model_then_parser() -> None:
    chain = app.build_chain(async_only_model(ANSWER))

    assert isinstance(chain, RunnableSequence)
    assert [type(step) for step in chain.steps] == [
        ChatPromptTemplate,
        AsyncOnlyModel,
        StrOutputParser,
    ]


def test_prompt_sends_a_system_and_a_human_message() -> None:
    messages = app.build_prompt().invoke({"pregunta": "¿Qué es LCEL?"}).to_messages()

    assert messages == [
        SystemMessage(content=app.SYSTEM_PROMPT),
        HumanMessage(content="¿Qué es LCEL?"),
    ]


def test_prompt_rejects_a_dict_without_the_pregunta_key() -> None:
    with pytest.raises(KeyError, match="pregunta"):
        app.build_prompt().invoke({"question": "¿Qué es LCEL?"})


async def test_main_runs_the_chain_asynchronously(
    api_key: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(app, "build_model", lambda: async_only_model(ANSWER))

    assert await app.main(["¿Qué es LCEL?"]) == 0
    assert capsys.readouterr().out == ANSWER + "\n"


async def test_main_sends_the_question_to_openai_and_prints_plain_text(
    fake_openai: FakeOpenAI, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_openai.answer(ANSWER)

    assert await app.main(["¿Qué", "es", "LCEL?"]) == 0

    assert capsys.readouterr().out == ANSWER + "\n"
    [request] = fake_openai.requests
    assert (request["model"], request["temperature"]) == ("gpt-4o-mini", app.TEMPERATURE)
    assert request["messages"] == [
        {"role": "system", "content": app.SYSTEM_PROMPT},
        {"role": "user", "content": "¿Qué es LCEL?"},
    ]


async def test_main_asks_the_default_question_without_arguments(
    fake_openai: FakeOpenAI,
) -> None:
    fake_openai.answer(ANSWER)

    assert await app.main([]) == 0

    assert fake_openai.requests[0]["messages"][-1]["content"] == app.DEFAULT_QUESTION


async def test_main_uses_the_model_from_the_environment(
    fake_openai: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    fake_openai.answer(ANSWER)

    assert await app.main([]) == 0

    assert fake_openai.requests[0]["model"] == "gpt-test"


async def test_main_reports_an_api_failure_on_stderr_without_a_traceback(
    fake_openai: FakeOpenAI, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_openai.fail(401, "Incorrect API key provided")

    assert await app.main([]) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("OpenAI no pudo responder (OpenAIAuthenticationError): ")
    assert "Incorrect API key provided" in output.err
    assert len(fake_openai.requests) == 1


async def test_main_without_a_key_explains_how_to_configure_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert await app.main([]) == 1
    assert ".env.example" in capsys.readouterr().err
