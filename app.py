"""Streamlit UI: AI-помощник менеджера продаж AmoCRM (O-complex)."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import streamlit as st
from openai import OpenAI

from models import (
    ManagerAssistantResponse,
    client_reply_echoes_question,
    get_response_format_instruction,
)

ROOT = Path(__file__).resolve().parent
CRM_PATH = ROOT / "crm_mocks.json"
KB_PATH = ROOT / "knowledge_base.md"

# Подхватываем .env без обязательного python-dotenv
_env_file = ROOT / ".env"
if _env_file.is_file():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@st.cache_data
def load_crm_mocks() -> list[dict]:
    with CRM_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_knowledge_base() -> str:
    return KB_PATH.read_text(encoding="utf-8")


def llm_config() -> dict[str, str]:
    base = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://127.0.0.1:11434/v1"
    )
    return {
        "base_url": base.rstrip("/"),
        "api_key": os.getenv("OPENAI_API_KEY") or "ollama",
        "model": os.getenv("OLLAMA_MODEL") or os.getenv("OPENAI_MODEL") or "qwen3:8b",
    }


def find_client(clients: list[dict], client_id: str) -> dict | None:
    for c in clients:
        if c["id"] == client_id:
            return c
    return None


def history_to_chat(history: list[dict]) -> list[dict]:
    """CRM history → сообщения чата (клиент / менеджер)."""
    messages: list[dict] = []
    for item in history:
        role = "client" if item.get("type") == "входящее" else "manager"
        channel = item.get("channel", "")
        date = item.get("date", "")
        prefix = f"[{date} · {channel}] " if date or channel else ""
        messages.append({"role": role, "content": prefix + item.get("summary", "")})
    return messages


def init_client_session(client: dict) -> None:
    st.session_state.selected_client_id = client["id"]
    st.session_state.chat_messages = history_to_chat(client.get("history") or [])
    st.session_state.client_reply = ""
    st.session_state.manager_hint = ""
    st.session_state.last_error = ""
    st.session_state.message_draft = ""
    st.session_state.manager_comment = ""


def copy_reply_to_draft() -> None:
    """Подставить client_reply в поле ввода чата (до создания виджета на следующем run)."""
    reply = (st.session_state.get("client_reply") or "").strip()
    st.session_state.message_draft = reply
    st.session_state.compose_role = "manager"


def last_client_text(chat: list[dict]) -> str:
    for message in reversed(chat):
        if message.get("role") == "client":
            return message.get("content") or ""
    return ""


def build_prompt(kb: str, client: dict, chat: list[dict]) -> list[dict]:
    profile = {
        "id": client.get("id"),
        "name": client.get("name"),
        "company": client.get("company"),
        "role": client.get("role"),
        "city": client.get("city"),
        "funnel_status": client.get("funnel_status"),
        "budget": client.get("budget"),
        "interest": client.get("interest"),
        "tags": client.get("tags"),
    }
    transcript = "\n".join(
        f"{'Клиент' if m['role'] == 'client' else 'Менеджер'}: {m['content']}"
        for m in chat
    )
    system = (
        "Ты AI-помощник менеджера продаж O-complex в AmoCRM.\n"
        "По базе знаний, профилю клиента и переписке подготовь ответ клиенту "
        "и внутреннюю подсказку менеджеру по допродажам.\n\n"
        f"{get_response_format_instruction()}\n\n"
        "=== База знаний ===\n"
        f"{kb}"
    )
    question = last_client_text(chat)
    user = (
        "=== Профиль клиента ===\n"
        f"{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
        "=== Переписка ===\n"
        f"{transcript or '(пусто)'}\n\n"
        "=== На что ответить ===\n"
        "Последнее сообщение клиента (на него нужен ответ, его нельзя копировать):\n"
        f"{question or '(нет)'}\n\n"
        "Сформируй JSON с полями client_reply и manager_hint.\n"
        "client_reply — вежливый ответ по базе знаний. "
        "Если точного правила нет, честно напиши, что в базе знаний такого условия нет, "
        "и не выдумывай скидку.\n"
        "manager_hint — текст шпаргалки без JSON и без подписей полей вроде action."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


# Если модель дважды повторила вопрос, в карточку кладём честный отказ, а не цитату.
HONEST_UNKNOWN_REPLY = (
    "В базе знаний нет подтверждённого правила под этот вопрос, "
    "поэтому не буду обещать скидку. Уточню и вернусь с точным ответом."
)


def _complete(openai_client: OpenAI, model: str, messages: list[dict]) -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("Модель вернула пустой ответ.")
    return raw


def _parse_completion(raw: str) -> ManagerAssistantResponse:
    return ManagerAssistantResponse.parse_ai_json(extract_json_text(raw))


def call_llm(kb: str, client: dict, chat: list[dict]) -> ManagerAssistantResponse:
    cfg = llm_config()
    openai_client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=90.0)
    messages = build_prompt(kb, client, chat)
    client_messages = [m["content"] for m in chat if m.get("role") == "client"]
    raw = _complete(openai_client, cfg["model"], messages)
    result = _parse_completion(raw)
    if not client_reply_echoes_question(result.client_reply, client_messages):
        return result

    question = last_client_text(chat)
    messages = [
        *messages,
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "client_reply повторил вопрос клиента. Это ошибка.\n"
                f"Вопрос клиента: {question}\n"
                "Напиши вежливый ОТВЕТ по базе знаний. Не цитируй и не перефразируй вопрос.\n"
                "Если точного правила в базе нет — честно напиши, что такого условия нет, "
                "и не выдумывай скидку.\n"
                "manager_hint — обычный текст без ключей вроде action.\n"
                "Верни только JSON с полями client_reply и manager_hint."
            ),
        },
    ]
    try:
        retried = _parse_completion(_complete(openai_client, cfg["model"], messages))
    except Exception:
        return result.model_copy(update={"client_reply": HONEST_UNKNOWN_REPLY})
    if client_reply_echoes_question(retried.client_reply, client_messages):
        return retried.model_copy(update={"client_reply": HONEST_UNKNOWN_REPLY})
    return retried


def render_cards(reply: str, hint: str) -> None:
    placeholder = (
        "<span style='color:#666'>Пока пусто — нажмите «Получить подсказку».</span>"
    )
    reply_html = html.escape(reply) if reply else placeholder
    hint_html = html.escape(hint) if hint else placeholder
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(
            f"""
<div style="
  border: 1px solid #2e7d32;
  border-left: 5px solid #2e7d32;
  background: #e8f5e9;
  border-radius: 10px;
  padding: 1rem 1.1rem;
  min-height: 160px;
">
  <div style="font-weight: 700; color: #1b5e20; margin-bottom: 0.5rem;">
    ✉ Ответ клиенту
  </div>
  <div style="color: #1a1a1a; white-space: pre-wrap; line-height: 1.45;">
    {reply_html}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin-top: 0.75rem;'></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "📋 Вставить в поле ввода",
            key="copy_reply_to_input",
            use_container_width=True,
            disabled=not bool(reply.strip()),
            on_click=copy_reply_to_draft,
            help="Подставить ответ AI в поле «Новое сообщение» (роль: Менеджер)",
        )
    with col2:
        st.markdown(
            f"""
<div style="
  border: 1px solid #ef6c00;
  border-left: 5px solid #ef6c00;
  background: #fff3e0;
  border-radius: 10px;
  padding: 1rem 1.1rem;
  min-height: 160px;
">
  <div style="font-weight: 700; color: #e65100; margin-bottom: 0.5rem;">
    💡 Подсказка менеджеру
  </div>
  <div style="color: #1a1a1a; white-space: pre-wrap; line-height: 1.45;">
    {hint_html}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="O-complex · AI помощник AmoCRM",
        page_icon="💬",
        layout="wide",
    )

    clients = load_crm_mocks()
    kb = load_knowledge_base()

    if "selected_client_id" not in st.session_state:
        init_client_session(clients[0])
    if "chat_messages" not in st.session_state:
        client0 = find_client(clients, st.session_state.selected_client_id) or clients[0]
        init_client_session(client0)
    st.session_state.setdefault("message_draft", "")
    st.session_state.setdefault("manager_comment", "")

    # —— Сайдбар: выбор клиента ——
    with st.sidebar:
        st.title("Клиенты CRM")
        st.caption("Моковые лиды из AmoCRM")

        options = {c["id"]: c for c in clients}
        labels = {
            c["id"]: f"{c['name']} · {c.get('funnel_status', '—')}" for c in clients
        }
        selected_id = st.radio(
            "Выберите клиента",
            options=list(options.keys()),
            format_func=lambda cid: labels[cid],
            index=list(options.keys()).index(st.session_state.selected_client_id)
            if st.session_state.selected_client_id in options
            else 0,
            label_visibility="collapsed",
        )

        if selected_id != st.session_state.selected_client_id:
            init_client_session(options[selected_id])
            st.rerun()

        client = options[selected_id]
        st.divider()
        st.subheader(client["name"])
        st.markdown(f"**Статус воронки:** {client.get('funnel_status', '—')}")
        st.markdown(f"**Бюджет:** {client.get('budget', '—')}")
        if client.get("company"):
            st.markdown(f"**Компания:** {client['company']}")
        if client.get("role"):
            st.markdown(f"**Роль:** {client['role']}")
        if client.get("city"):
            st.markdown(f"**Город:** {client['city']}")
        if client.get("interest"):
            st.markdown("**Интерес:** " + ", ".join(client["interest"]))
        if client.get("tags"):
            st.markdown("**Теги:** " + ", ".join(f"`{t}`" for t in client["tags"]))

        st.divider()
        st.markdown("#### История обращений")
        for item in client.get("history") or []:
            st.markdown(
                f"**{item.get('date', '—')}** · {item.get('channel', '')} "
                f"({item.get('type', '')})\n\n"
                f"{item.get('summary', '')}"
            )

        cfg = llm_config()
        st.divider()
        st.caption(f"Модель: `{cfg['model']}`\n\nEndpoint: `{cfg['base_url']}`")

    # —— Центр: чат ——
    st.title("Чат AmoCRM")
    st.caption(
        "Переписка с клиентом. Введите новое сообщение клиента (или черновик менеджера) "
        "и запросите подсказку AI."
    )

    chat_box = st.container(border=True)
    with chat_box:
        if not st.session_state.chat_messages:
            st.info("История пуста. Добавьте сообщение клиента ниже.")
        for msg in st.session_state.chat_messages:
            if msg["role"] == "client":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**Клиент**\n\n{msg['content']}")
            else:
                with st.chat_message("assistant", avatar="🧑‍💼"):
                    st.markdown(f"**Менеджер**\n\n{msg['content']}")

    role_label = st.radio(
        "Кто пишет сообщение",
        options=["client", "manager"],
        format_func=lambda r: "Клиент (запрос для подсказки)"
        if r == "client"
        else "Менеджер (черновик / реплика)",
        horizontal=True,
        key="compose_role",
    )

    with st.form("compose_form", clear_on_submit=True):
        text = st.text_area(
            "Новое сообщение",
            height=100,
            key="message_draft",
            placeholder="Например: «А есть скидка на набор Антиаллергия при заказе от 30 тыс.?»",
        )
        submitted = st.form_submit_button("Добавить в переписку", use_container_width=True)
        if submitted and text.strip():
            st.session_state.chat_messages.append(
                {"role": role_label, "content": text.strip()}
            )
            st.rerun()

    col_a, col_b = st.columns([2, 1])
    with col_a:
        get_hint = st.button("Получить подсказку", type="primary", use_container_width=True)
    with col_b:
        if st.button("Очистить карточки", use_container_width=True):
            st.session_state.client_reply = ""
            st.session_state.manager_hint = ""
            st.session_state.manager_comment = ""
            st.session_state.last_error = ""
            st.rerun()

    if get_hint:
        if not st.session_state.chat_messages:
            st.session_state.last_error = "Добавьте хотя бы одно сообщение в переписку."
        else:
            with st.spinner("Запрос к модели…"):
                try:
                    result = call_llm(kb, client, st.session_state.chat_messages)
                    st.session_state.client_reply = result.client_reply
                    st.session_state.manager_hint = result.manager_hint
                    st.session_state.last_error = ""
                except Exception as exc:  # noqa: BLE001 — UI не должен падать
                    st.session_state.last_error = (
                        "Не удалось получить ответ от LLM. "
                        "Проверьте, что Ollama запущена и модель скачана.\n\n"
                        f"Детали: {exc}"
                    )

    if st.session_state.get("last_error"):
        st.warning(st.session_state.last_error)

    st.subheader("Результаты AI")
    render_cards(
        st.session_state.get("client_reply", ""),
        st.session_state.get("manager_hint", ""),
    )

    st.markdown("##### Комментарий менеджера")
    st.text_area(
        "Комментарий менеджера",
        height=100,
        key="manager_comment",
        placeholder="Например: слишком длинно; не упоминать скидку; добавить срок доставки…",
        label_visibility="collapsed",
    )


if __name__ == "__main__":
    main()
