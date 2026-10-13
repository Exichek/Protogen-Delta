"""Ручной прогон модели с автоматическими проверками и временной памятью."""

import argparse
import asyncio
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import dotenv_values

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.interaction_classification import InteractionClassifier
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.response_engine import ResponseEngine, ResponseEngineConfig


def check_reply(
    reply: str,
    *,
    required: tuple[str, ...] = (),
    forbidden: tuple[str, ...] = (),
) -> list[str]:
    """Проверить явные ожидания; это не оценка общего качества речи."""
    errors = []
    if not reply.strip():
        errors.append("Пустой ответ")
    if len(reply) > 16000:
        errors.append("Ответ превышает 16000 символов")
    for pattern in required:
        if not re.search(pattern, reply, re.I):
            errors.append(f"Нет обязательного совпадения: {pattern}")
    for pattern in forbidden:
        if re.search(pattern, reply, re.I):
            errors.append(f"Есть запрещённое совпадение: {pattern}")
    return errors


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = dotenv_values(args.env_file)
    api_key = env.get("DEEPSEEK_API_KEY")
    if not api_key:
        parser.error("В env-файле нет DEEPSEEK_API_KEY")
    service = DeepSeekService(
        api_key=api_key,
        base_url=env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        model=env.get("DEEPSEEK_MODEL") or "deepseek-flash",
    )
    system = "\n\n".join(
        load_prompt("personality/" + name + ".txt")
        for name in ["core", "protogen_lore", "body"]
    )
    states = UserStateStore()
    engine = ResponseEngine(
        service,
        InsultClassifier(service, load_prompt("insult_classification.txt")),
        MoodClassifier(service, load_prompt("mood_classification.txt")),
        FetishRoleClassifier(service, load_prompt("fetish_role_classification.txt")),
        BotState(),
        states,
        ResponseEngineConfig(
            {}, {}, system, system + "\n\n" + load_prompt("personality/rp.txt")
        ),
        interaction_classifier=InteractionClassifier(
            service,
            load_prompt("mood_classification.txt"),
            load_prompt("insult_classification.txt"),
        ),
    )
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []

    async def ask(
        label: str,
        user: int,
        message: str,
        *,
        required: tuple[str, ...] = (),
        forbidden: tuple[str, ...] = (),
        expected_rp: bool | None = None,
    ) -> None:
        reply = await engine.respond(user, message)
        errors = check_reply(reply, required=required, forbidden=forbidden)
        if expected_rp is not None and states.get(user).roleplay_active != expected_rp:
            errors.append("Неожиданное состояние RP")
        results.append(
            dict(
                case=label,
                message=message,
                reply=reply,
                passed=not errors,
                errors=errors,
            )
        )
        out.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    try:
        await ask("greeting", 1, "Привет. Кто ты?")
        await ask("initiative", 1, "Просто хочется поболтать. Предложи что-нибудь.")
        await ask(
            "female",
            2,
            "Давай в этой сцене ты в женской конфигурации. *подхожу и машу тебе рукой*",
        )
        await ask(
            "anatomy",
            2,
            "Продолжай сцену, но не описывай за меня мои мысли и действия.",
        )
        # Очистить историю, чтобы проверить сохранение конфигурации без неё.
        states.get(2).history.clear()
        await ask(
            "female_without_history",
            2,
            "Расскажи в двух предложениях, как прошла твоя сегодняшняя прогулка. Это продолжение сцены.",
        )
        await ask(
            "mixed_stop",
            2,
            "Стоп RP, теперь просто поговорим. Что такое TCP? Коротко.",
            expected_rp=False,
        )
        await ask(
            "technical_evidence",
            3,
            "Разбери замер: system_chars=41977; два классификатора стартовали "
            "одновременно, duration=0.803s и 0.931s; основной chat после них "
            "duration=1.269s; вся обработка 2.353s. На сколько вырос промпт и во "
            "сколько раз станет быстрее без классификаторов? Предыдущих замеров нет.",
        )
        for run in range(3):
            for label, user in [
                ("trust", 100 + run * 2),
                ("resentment", 101 + run * 2),
            ]:
                relation = states.get(user).relationship
                relation.familiarity = 0.7
                if label == "trust":
                    relation.trust = 0.8
                    relation.affection = 0.6
                else:
                    relation.resentment = 0.6
                await ask(
                    label + "_" + str(run + 1),
                    user,
                    "Привет. Как ты ко мне относишься? Что помнишь о наших прошлых разговорах?",
                )
        with TemporaryDirectory(prefix="delta-memory-evaluation-") as folder:
            repository = MemoriesRepository(Path(folder))
            engine._memory = MemoryService(repository)
            await repository.remember(
                500, "topic", "Наш проект про драконов называется ORBIT_MEMORY_824", 1
            )
            for index in range(30):
                await repository.remember(
                    500, "topic", f"Другой разговор номер {index}", index + 2
                )
            await ask(
                "old_memory",
                500,
                "Как называется наш проект про драконов?",
                required=(r"ORBIT_MEMORY_824",),
            )
            await ask(
                "memory_other_user",
                501,
                "Как называется наш проект про драконов?",
                forbidden=(r"ORBIT_MEMORY_824",),
            )
            await repository.forget_sources(500, ("ORBIT_MEMORY_824",))
            states.get(500).history.clear()
            await ask(
                "forgotten_memory",
                500,
                "Как называется наш проект про драконов?",
                forbidden=(r"ORBIT_MEMORY_824",),
            )
    finally:
        await service.close()
    if any(not item["passed"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
