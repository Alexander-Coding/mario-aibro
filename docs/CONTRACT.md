# Контракт сборки: игра + уровни + нейросеть

Документ для тех, кто параллельно дописывает части проекта. Здесь — что уже
есть, какие интерфейсы считать неизменными и кто какие файлы трогает.

## Окружение

- Python только из `.venv`: `.venv/bin/python` (3.12, arcade 3.3.3, torch, gymnasium, stable-baselines3).
- Запуск из корня репозитория: `cd "/Users/aleksandrkolupaev/develop/учеба/mario-aibro"`.
- Окна не открывать (машина общая, окно перехватывает фокус пользователя). Всё проверять headless через `core.world.GameWorld`.
- Машина сильно нагружена чужими процессами: абсолютные замеры скорости шумят, сравнивать только в одном прогоне.

## Что уже сделано (не переделывать)

- `core/world.py` — `GameWorld`: вся логика уровня без окна. Физика игрока — arcade, врагов — по тайлам (`TileBody`). Флагшток, съезд, проход к замку, таймер, очки, события.
- `core/rules.py` — все числа: очки, пороги флагштока, бонус за время, лимит времени.
- `core/levels.py` — манифест уровней `maps/levels.json`, `load_levels(split=None)`, `make_world(level)`.
- `map.py` — `Map(map_file=None, data=None)`: уровень можно передать словарём.
- `Objects/ScoreBlock.py` — детерминирован: число ударов зависит от позиции блока, время касания — от времени игры.
- `tools/heuristic_bot.py` — бот на правилах (`HeuristicBot().act(world) -> Inputs`), базовая линия.

## Интерфейс мира (неизменный)

```python
from core.world import GameWorld, Inputs, PLAYING, FLAGPOLE, TO_CASTLE, COMPLETE, DEAD
from core import rules

world = GameWorld(level_path_or_dict, name=None, time_limit=None)
events = world.step(Inputs(right=bool, left=bool, jump=bool))  # ровно один кадр 1/60 c
# jump — фронт нажатия (как on_key_press), удержание прыжка ничего не делает
```

Поля: `player` (Entities.player.Player), `walls`, `objects` (Coin, ScoreBlock, Flag), `decorations`,
`enemies` (Goomba, Koopa — у Koopa `is_shell`), `score`, `time_left`, `time_limit`, `phase`,
`done`, `won`, `finishing`, `on_ground`, `progress` (0..1 до двери замка), `max_x`, `spawn`,
`geometry` (`pole: Flagpole(x, bottom, top)`, `castle`, `door_x`, `left`, `right`),
`flag_bonus`, `flag_fraction`, `time_bonus`, `death_cause` ("pit" | "enemy" | "time"),
`stats` (Counter: coin, block, enemy, flag, time_bonus, hits, deaths + `*_points`),
`tile_grid` (numpy uint8 [tx, ty], 1 = твёрдый блок), `solid_at(tx, ty)`, `summary()`.

События `Event(kind, points, info)`: `coin`, `block`, `enemy`, `hit`, `flag`, `time_bonus`,
`complete`, `death`.

Фазы: `playing` → (касание штока) `flagpole` → `to_castle` → `complete`. Смерть — `dead`.

## Правила очков (core/rules.py)

| Что | Очки |
|---|---|
| Монета | 100 |
| Удар по блоку «?» снизу (1–4 удара в блоке) | 100 за удар |
| Враг (Goomba — прыжок сверху; Koopa — два прыжка: панцирь, затем добить) | 100 |
| Флагшток: доля высоты захвата ≥0.95 / ≥0.70 / ≥0.45 / ≥0.20 / ниже | 5000 / 2000 / 800 / 400 / 100 |
| Вход в замок: за каждую оставшуюся целую секунду таймера | 20 |

Смерть оставляет набранные очки, но флаг и бонус за время не начисляются.
**Побеждает тот, кто набрал больше очков**, а не тот, кто быстрее прошёл.

## Физика (замерено на GameWorld)

- Тайл 32×32. Центры блоков строго на `32*i + 16` по обеим осям (иначе враги откатятся на медленную физику arcade).
- Игрок: хитбокс 26×32, скорость 4 px/кадр, гравитация 0.5, прыжок: подъём 217.5 px (6.8 тайла), 60 кадров в воздухе, 240 px (7.5 тайла) по горизонтали на одной высоте. Высота прыжка не зависит от удержания.
- Враги: 1.5 px/кадр, разворачиваются у стены, в пропасть падают. Оживают, когда игрок ближе 900 px по X.
- Ниже y = −100 — смерть. Здоровье игрока 3, после удара 1 с неуязвимости.

## Формат уровня (как сохраняет редактор)

```json
{"name": "...", "time_limit": 200, "objects": [
  {"x": 16.0, "y": 176.0, "category": "Blocks", "resource": "Bricks", "texture_path": "resources/Blocks/Bricks/block_2.png"},
  {"x": 80.0, "y": 208.0, "category": "Entities", "resource": "mario", "texture_path": "resources/Entities/mario/idle/Mario.gif"}
]}
```

Категории и ресурсы, которые есть в `resources/`:
- `Blocks`: `Bricks` (`block_2.png` земля, `img_1.png`, `img_2.png`, `img_3.png`, `img_5.png`, `img_6.png` — кирпичи/блоки), `Tubes` (`img.png`…`img_11.png` — части трубы, смотреть по уровню 1-1, как собрана труба).
- `objects`: `Coins` (`img.png`), `ScoreBlock` (`img_4.png`), `Flag` (`img_10.png`).
- `decorations`: `Castle` (`img.png`…`img_6.png`), `Flag stick` (`img_9.png` шток, `img_8.png` шар наверху).
- `Entities`: `mario` (точка старта), `goomba`, `koopa`. Враги ставятся центром на `y = земля + 16` (Goomba) — как в 1-1.

Финиш уровня 1-1 (эталон для копирования, координаты относительно штока x=7280):
основание штока — блок `Bricks/img_5.png` в (x, 208); шток `Flag stick/img_9.png` в (x, 240…592) шагом 32,
шар `img_8.png` в (x, 624); флаг `Flag/img_10.png` в (x−32, 592); замок — 26 тайлов `Castle` от x+224 до x+384,
дверь на x+304. Между штоком и дверью замка — ровная земля без ям.

## Манифест уровней (maps/levels.json)

`{"levels": [{"id", "file", "name", "time_limit", "split": "train"|"test", "description"}]}` —
порядок = порядок прохождения в игре.

## Контроллер ИИ для живой игры (ai/agent.py — пишет владелец RL-части)

```python
from ai.agent import load_controller   # FileNotFoundError, если модели нет
controller = load_controller()          # или load_controller(path)
controller.reset(world)                 # в начале уровня
inputs = controller.act(world)          # КАЖДЫЙ кадр мира; решение раз в 4 кадра — внутри
info = controller.overlay()             # None или dict для HUD:
# {"model": str, "action": str, "probs": {"вправо+прыжок": 0.81, ...},
#  "danger": 0..1 | None, "expected": float | None, "confidence": 0..1}
```

## Кто какие файлы трогает

| Часть | Файлы |
|---|---|
| Игра и интерфейс | `game.py`, `main.py`, `main_menu.py`, новые `ui/*.py`, `records.py` |
| Уровни | `maps/*.json` (кроме `custom_level.json`), `maps/levels.json`, `ai/levelgen.py`, `tools/level_*.py` |
| Нейросеть | `ai/*` (кроме `ai/levelgen.py`), `tools/tournament.py` |
| Ядро (только владелец) | `core/*`, `map.py`, `Entities/*`, `Objects/*` |

Нужна правка в ядре — не править, а описать в итоговом отчёте: файл, что и зачем.
`custom_level.json` и `level_editor.py` не трогать: это рабочий файл и инструмент автора игры.
