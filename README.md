# pogo

Голосовой ассистент в терминале. Речь распознаётся локально (whisper.cpp), «мозг» и синтез речи подключаемые.

```
микрофон → RealtimeSTT (Silero VAD + whisper.cpp) → brain → RealtimeTTS (Silero / Edge / Piper / say)
```

## Запуск

```bash
brew install portaudio
uv sync
uv run python -m pogo                    # агент pi с deepseek/deepseek-v4-flash (полный доступ к инструментам)
uv run python -m pogo --brain deepseek   # просто чат с DeepSeek, нужен DEEPSEEK_API_KEY
```

Пробел — начать/остановить разговор, Esc — перебить ответ, q — выход.
Пока ассистент говорит, микрофон выключен, чтобы он не слышал сам себя.

Опции: `--model` (модель для brain), `--whisper small|medium|large-v3-turbo`, `--tts`, `--voice`, `--debug`, `--record [PATH]`.

`--record` пишет весь диалог в один WAV так, как он звучал: ваш голос с микрофона и голос ассистента
на общей шкале времени, с реальными паузами (ожидание ответа, работа инструментов). Файл сохраняется
при выходе по `q`, по умолчанию в `~/Downloads/`.

## Синтез речи

| `--tts` | Голоса (`--voice`) | Где | Синтез фразы | Заметки |
|---|---|---|---|---|
| `silero` (по умолчанию) | xenia, baya, kseniya, aidar, eugene | локально, CPU | ~0,1 с | лицензия некоммерческая; цифры и латиницу не читает (промпт просит писать их словами) |
| `edge` | ru-RU-SvetlanaNeural, ru-RU-DmitryNeural | облако Microsoft | 1–3 с | самые естественные; неофициальный доступ, текст уходит в Microsoft |
| `piper` | ru_RU-irina/denis/dmitri/ruslan-medium | локально | ~0,3 с | синтетичнее Silero |
| `say` | Milena | локально | ~0,7 с | механический; лучше скачать Milena (Enhanced) в настройках «Устный контент» |

Модели скачиваются при первом запуске в `~/.pogo/models/`.

Сравнить на слух — одна фраза всеми движками и голосами в `~/Downloads/pogo-tts-samples/`:

```bash
uv run python -m pogo.samples                     # все движки
uv run python -m pogo.samples --engines silero --text "Своя фраза"
```

## Логи

Каждая сессия пишет `~/.pogo/logs/pogo-<время>.log` (путь печатается при старте). В терминал логи не выводятся.

- `turn=N +X.XXs <этап>` — хронология хода от начала вашей речи: `speech_end`, `transcribed`, `prompt_sent`,
  `first_text`, `tool`, `sentence_synth_start/end`, `audio_start`, `audio_end`, `done`.
- `turn=N summary` — задержки хода (то же видно в терминале после ответа):
  `stt` (распознавание), `llm_first_token`, `tts_first_audio`, `voice_to_voice` (от конца вашей фразы до первого звука), `tools`, `total`.
- `pogo.tts: synthesized …` — время синтеза и длина аудио на каждое предложение; пустое аудио — `ERROR`.
- `pogo.brain` — запросы, вызовы инструментов с длительностью, токены, ретраи, stderr pi.
- `answer has … but no audio was played` — ответ был, а звука не было.
- `--debug` добавляет все события pi и смены статуса.

```bash
grep summary ~/.pogo/logs/pogo-*.log | tail     # задержки по ходам
grep -E 'ERROR|WARNING' ~/.pogo/logs/pogo-*.log  # инциденты
```

## Brains

`pogo/brains.py`: brain отдаёт поток событий `("text", …)`, `("tool", …)`, `("error", …)` и умеет `abort()`.
Новый бэкенд (Claude, Codex, …) — класс с `ask`/`abort`/`close` и строка в `BRAINS`.
Новый TTS — наследник `PcmEngine` с методом `render(text) -> PCM` и строка в `TTS_ENGINES` (`pogo/tts.py`).
