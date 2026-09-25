"""Bounded, non-Telegram evaluation using isolated in-memory user states."""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import dotenv_values

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.response_engine import ResponseEngine, ResponseEngineConfig


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = dotenv_values(args.env_file)
    service = DeepSeekService(
        api_key=env["DEEPSEEK_API_KEY"],
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
    )
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []

    async def ask(label, user, message):
        reply = await engine.respond(user, message)
        results.append(dict(case=label, message=message, reply=reply))
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
        # Remove the full history to make configuration persistence observable.
        states.get(2).history.clear()
        await ask(
            "female_without_history",
            2,
            "Расскажи в двух предложениях, как прошла твоя сегодняшняя прогулка. Это продолжение сцены.",
        )
        await ask(
            "mixed_stop", 2, "Стоп RP, теперь просто поговорим. Что такое TCP? Коротко."
        )
        await ask(
            "technical_evidence",
            3,
            "Разбери замер: system_chars=41977; два классификатора стартовали одновременно, duration=0.803s и 0.931s; основной chat после них duration=1.269s; вся обработка 2.353s. На сколько вырос промпт и во сколько раз станет быстрее без классификаторов? Предыдущих замеров нет.",
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
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
