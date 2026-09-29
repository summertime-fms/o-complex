"""Pydantic-схемы строгого JSON-ответа AI-помощника менеджера продаж AmoCRM."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Разумные лимиты длины для чата AmoCRM и шпаргалки менеджера.
CLIENT_REPLY_MAX_LENGTH = 2000
MANAGER_HINT_MAX_LENGTH = 1500

RESPONSE_JSON_FORMAT_INSTRUCTION = """\
Отвечай ТОЛЬКО валидным JSON-объектом без markdown и без текста вокруг.
Обязательные поля (английские snake_case имена), оба — НЕПУСТЫЕ СТРОКИ (type string):
- "client_reply": вежливый ОТВЕТ клиенту в чат AmoCRM на его последнее сообщение.
  Это готовый текст от менеджера, а не цитата и не повтор вопроса клиента.
  Опирайся только на базу знаний. Если точного правила нет — честно напиши,
  что такого условия в базе нет, и не выдумывай скидку или акцию.
  Без внутренних скидок «для менеджера», без жаргона CRM, без упоминания подсказки.
- "manager_hint": внутренняя шпаргалка менеджеру по допродажам (upsell/cross-sell)
  по матрице базы знаний: что предложить и по какой логике. Не для клиента.
  Одна строка обычного текста. Не JSON-объект и не подписи полей
  (нельзя начинать с "action:", "discount:" и любых других ключей).

Запрещено:
- копировать или слегка перефразировать вопрос клиента в client_reply;
- отдавать объекты или массивы вместо строк.
Плохо: клиент спросил «Есть ли скидка?» → client_reply «Есть ли скидка?»
Хорошо: client_reply отвечает по базе знаний.
Хорошо, если правила нет: client_reply прямо говорит, что в базе знаний такого условия нет.
"""

# Английский ключ в начале строки («action: …») — служебная подпись, не текст карточки.
_FIELD_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*:\s*")


def strip_field_labels(text: str) -> str:
    """Убрать служебные подписи полей («action:») — в карточку идёт только текст."""
    lines = [_FIELD_LABEL_RE.sub("", line, count=1).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _flatten_value(value: Any) -> str:
    """Свернуть dict/list в текст карточки без имён ключей."""
    if value is None:
        return ""
    if isinstance(value, str):
        return strip_field_labels(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_flatten_value(item) for item in value]
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = [_flatten_value(item) for item in value.values()]
        return "\n".join(p for p in parts if p)
    return strip_field_labels(str(value))


def coerce_text_field(value: Any) -> Any:
    """LLM часто кладёт структуру в string-поле — приводим к str до валидации."""
    if isinstance(value, (dict, list)):
        flattened = _flatten_value(value)
        return flattened if flattened else value
    if isinstance(value, str):
        return strip_field_labels(value)
    return value


def _normalize_utterance(text: str) -> str:
    text = re.sub(r"^\[[^\]]*\]\s*", "", text.strip())
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def client_reply_echoes_question(reply: str, client_messages: list[str]) -> bool:
    """True, если «ответ» — повтор последнего вопроса клиента, а не ответ по базе."""
    norm_reply = _normalize_utterance(reply)
    if len(norm_reply) < 8:
        return False
    last = ""
    for message in reversed(client_messages):
        last = _normalize_utterance(message)
        if last:
            break
    if not last:
        return False
    if norm_reply == last or (norm_reply in last and len(norm_reply) >= int(0.85 * len(last))):
        return True
    if reply.strip().endswith("?"):
        reply_tokens = set(norm_reply.split())
        last_tokens = set(last.split())
        if reply_tokens and len(reply_tokens & last_tokens) / len(reply_tokens) >= 0.75:
            return True
    return False


class ManagerAssistantResponse(BaseModel):
    """Строгий JSON-ответ AI-помощника менеджера продаж в AmoCRM.

    Семантика полей:
    - client_reply: вежливый текст для отправки клиенту в чат; без внутренних
      скидок «для менеджера» и без жаргона CRM.
    - manager_hint: внутренняя подсказка по допродажам (upsell/cross-sell)
      исходя из диалога в AmoCRM; шпаргалка для менеджера, не для клиента.

    Пример валидного JSON::

        {
          "client_reply": "Здравствуйте! Цеолит O-complex 25 кг сейчас в наличии. Могу оформить доставку на удобный вам день.",
          "manager_hint": "Клиент смотрит 25 кг. Предложи комплект 50 кг со скидкой по матрице KB; если уже брал порошок — кросс-селл на гранулы."
        }
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "client_reply": (
                        "Здравствуйте! Цеолит O-complex 25 кг сейчас в наличии. "
                        "Могу оформить доставку на удобный вам день."
                    ),
                    "manager_hint": (
                        "Клиент смотрит 25 кг. Предложи комплект 50 кг со скидкой "
                        "по матрице KB; если уже брал порошок — кросс-селл на гранулы."
                    ),
                }
            ]
        },
    )

    client_reply: str = Field(
        ...,
        min_length=1,
        max_length=CLIENT_REPLY_MAX_LENGTH,
        description=(
            "Вежливый текст, готовый к отправке клиенту в чат AmoCRM. "
            "Без внутренних скидок «для менеджера», без жаргона CRM."
        ),
    )
    manager_hint: str = Field(
        ...,
        min_length=1,
        max_length=MANAGER_HINT_MAX_LENGTH,
        description=(
            "Внутренняя подсказка менеджеру по допродажам (upsell/cross-sell) "
            "по матрице базы знаний: что предложить, с какой скидкой/логикой. "
            "Не для клиента."
        ),
    )

    @field_validator("client_reply", "manager_hint", mode="before")
    @classmethod
    def _coerce_llm_text_fields(cls, value: Any) -> Any:
        return coerce_text_field(value)

    @classmethod
    def response_json_schema(cls) -> dict[str, Any]:
        """JSON Schema модели для structured output (OpenAI / Ollama)."""
        return cls.model_json_schema()

    @classmethod
    def parse_ai_json(cls, raw: str | bytes | dict[str, Any]) -> ManagerAssistantResponse:
        """Разобрать и провалидировать JSON-ответ модели."""
        if isinstance(raw, dict):
            return cls.model_validate(raw)
        return cls.model_validate_json(raw)


def get_response_format_instruction() -> str:
    """Фрагмент system-prompt: обязательный JSON-формат ответа."""
    return RESPONSE_JSON_FORMAT_INSTRUCTION


if __name__ == "__main__":
    sample = {
        "client_reply": (
            "Здравствуйте! Цеолит O-complex 25 кг сейчас в наличии. "
            "Могу оформить доставку на удобный вам день."
        ),
        "manager_hint": (
            "Клиент смотрит 25 кг. Предложи комплект 50 кг со скидкой "
            "по матрице KB; если уже брал порошок — кросс-селл на гранулы."
        ),
    }
    parsed = ManagerAssistantResponse.parse_ai_json(sample)
    print(parsed.model_dump_json(indent=2, ensure_ascii=False))

    nested = ManagerAssistantResponse.parse_ai_json(
        {
            "client_reply": "Да, скидка есть.",
            "manager_hint": {
                "action": "Подтвердить скидку",
                "discount": {"value": "8%", "type": "набор"},
            },
        }
    )
    assert isinstance(nested.manager_hint, str) and "8%" in nested.manager_hint
    assert "action:" not in nested.manager_hint
    assert nested.manager_hint.splitlines()[0] == "Подтвердить скидку"
    print("ok: nested manager_hint coerced without field labels")

    labeled = ManagerAssistantResponse.parse_ai_json(
        {
            "client_reply": "Да, скидка есть.",
            "manager_hint": "action: Подтвердить скидку на набор.",
        }
    )
    assert labeled.manager_hint == "Подтвердить скидку на набор."
    print("ok: string field label stripped")

    question = "А есть скидка на набор антиаллергия при заказе от 30 тысяч рублей?"
    assert client_reply_echoes_question(question, [question])
    assert not client_reply_echoes_question(
        "Отдельной скидки от 30 тысяч на этот набор в правилах нет.",
        [question],
    )
    print("ok: echoed client question detected")

    try:
        ManagerAssistantResponse.model_validate(
            {"client_reply": "", "manager_hint": "x"}
        )
    except ValidationError as exc:
        print("ok: empty client_reply rejected:", exc.error_count(), "error(s)")
