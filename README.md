# Cadena LCEL asíncrona

Ejercicio **"Refactorización a LCEL Asíncrono"** (Módulo 2, Unidad 1) del curso AI Engineering de Coderhouse.

En el Módulo 1 se llamaba al SDK de OpenAI a mano: armar los mensajes, llamar al cliente, sacar el texto de la respuesta. Este script hace lo mismo con una cadena declarativa de LangChain:

```python
chain = build_prompt() | model | StrOutputParser()
respuesta = await chain.ainvoke({"pregunta": pregunta})
```

| Eslabón | Entrada → salida |
| --- | --- |
| `ChatPromptTemplate` con rol `system` y `human` | `{"pregunta": "..."}` → `ChatPromptValue` |
| `ChatOpenAI` (`gpt-4o-mini`, `temperature=0.2`) | `ChatPromptValue` → `AIMessage` |
| `StrOutputParser` | `AIMessage` → `str` |

## Instalación

Con [uv](https://docs.astral.sh/uv/) (instala Python 3.12 y las dependencias, incluidas las de desarrollo):

```bash
uv sync
```

Con pip, en un entorno virtual de Python 3.12:

```bash
pip install -r requirements-dev.txt   # requirements.txt si solo vas a correr el script
```

Se instalan los paquetes que pide el enunciado (`langchain`, `langchain-openai` y `python-dotenv`) más `langchain-core` y `openai`, que el código importa directamente. Las piezas de LCEL (`ChatPromptTemplate`, `StrOutputParser`, el operador `|`) se importan de `langchain-core`, que es donde viven en LangChain 1.x; el modelo, de `langchain-openai`.

## Configuración

```bash
cp .env.example .env
```

| Variable | Obligatoria | Para qué |
| --- | --- | --- |
| `OPENAI_API_KEY` | sí | API key de OpenAI |
| `OPENAI_MODEL` | no | Modelo a usar; por defecto `gpt-4o-mini` |

## Uso

```bash
uv run main.py "¿Para qué sirve StrOutputParser?"
```

Sin argumento, responde una pregunta de ejemplo. Imprime solo el texto de la respuesta, no el objeto `AIMessage`.

Si falta la API key, lo dice y explica cómo configurarla. Si OpenAI rechaza la llamada (key inválida, sin conexión, rate limit), imprime el motivo en una línea por stderr y termina con código 1, sin traceback:

```text
OpenAI no pudo responder (OpenAIAuthenticationError): Error code: 401 - Incorrect API key provided: ...
```

## Tests

```bash
uv run pytest
```

No necesitan API key ni internet:

- Las pruebas de la cadena y del prompt usan un modelo falso de LangChain que solo responde por la vía asíncrona. `test_main_runs_the_chain_asynchronously` corre `main` con ese modelo: si `main` usara `invoke()` en lugar de `await ainvoke()`, falla.
- Otras pruebas de `main` corren el `ChatOpenAI` real contra un servidor local que imita la API de OpenAI. Así verifican lo que viaja por la red (modelo, `temperature`, mensaje `system` y mensaje `user` con la pregunta), que se imprime texto plano y que un 401 se reporta sin traceback.

## Criterios de aceptación

| Criterio | Dónde se cumple |
| --- | --- |
| El flujo principal usa solo LCEL (`\|`) | `build_chain`: `build_prompt() \| model \| StrOutputParser()` |
| Ejecución asíncrona con `await chain.ainvoke(...)` | `main`, con la pregunta como diccionario `{"pregunta": ...}`; lo prueba `test_main_runs_the_chain_asynchronously` |
| Salida en texto plano mediante `StrOutputParser` | Último eslabón de la cadena; lo prueba `test_chain_returns_plain_text_not_a_message` |
| `ChatPromptTemplate` con roles System/Human | `build_prompt`; lo prueban `test_prompt_sends_a_system_and_a_human_message` y `test_main_sends_the_question_to_openai_and_prints_plain_text` |
| `temperature` y `model_name` configurados | `build_model`: `ChatOpenAI(model_name=..., temperature=0.2)` |
