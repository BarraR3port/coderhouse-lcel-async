import asyncio
import os
import sys

import openai
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
DEFAULT_QUESTION = "¿Qué gano usando LCEL en lugar de llamar al SDK de OpenAI a mano?"

SYSTEM_PROMPT = (
    "Sos un tutor de ingeniería de IA. Respondé en español, en tres oraciones como máximo "
    "y con un ejemplo concreto cuando ayude."
)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{pregunta}")])


def build_model() -> ChatOpenAI:
    return ChatOpenAI(model_name=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), temperature=TEMPERATURE)


def build_chain(model: BaseChatModel) -> Runnable[dict[str, str], str]:
    return build_prompt() | model | StrOutputParser()


async def main(argv: list[str]) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Falta OPENAI_API_KEY: copiá .env.example a .env y completala.", file=sys.stderr)
        return 1
    chain = build_chain(build_model())
    pregunta = " ".join(argv).strip() or DEFAULT_QUESTION
    try:
        respuesta = await chain.ainvoke({"pregunta": pregunta})
    except openai.APIError as error:
        failure = f"OpenAI no pudo responder ({type(error).__name__}): {error.message}"
        print(failure, file=sys.stderr)
        return 1
    print(respuesta)
    return 0


if __name__ == "__main__":
    load_dotenv()
    sys.exit(asyncio.run(main(sys.argv[1:])))
