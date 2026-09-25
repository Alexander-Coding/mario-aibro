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
- `Objects/ScoreBlock.py` — блок «?» твёрдый; число ударов (1–4) детерминировано позицией блока; `hit()` — удар снизу.
- `tools/heuristic_bot.py` — бот на правилах (`HeuristicBot().act(world) -> Inputs`), базовая линия.

## Интерфейс мира (неизменный)

```python
from core.world import (GameWorld, Inputs, PLAYING, FLAGPOLE, TO_CASTLE, COMPLETE, DEAD,
                        PIPE_DOWN, PIPE_SIDE, PIPE_UP, PIPE_PHASES)
from core import rules

world = GameWorld(level_path_or_dict, name=None, time_limit=None)
events = world.step(Inputs(right=bool, left=bool, jump=bool, down=bool))  # ровно один кадр 1/60 c
# jump — фронт нажатия (как on_key_press), удержание прыжка ничего не делает
# down — «вниз» удерживается: стоя на трубе-входе, уйти в секретную комнату
```

Уровень состоит из частей `Area`: сам уровень (`world.main`) и секретные комнаты (`world.areas[key]`).
`world.area` — часть, где сейчас игрок; `world.in_room` — игрок в комнате. Поля ниже всегда отдают
**текущую** часть: `walls` (стены, включая блоки «?»), `objects` (Coin, Flag — блоков «?» тут больше нет),
`blocks` (list[ScoreBlock]), `decorations`, `enemies`, `tile_grid`, `solid_at(tx, ty)`, `physics_engine`.
У `Area`: `key`, `kind` ("main" | "room"), `theme` ("overworld" | "underground"), `spawn`,
`left/right/bottom/top` (границы для камеры), `warps` (трубы-входы), `exit` (устье выхода из комнаты).

Прочие поля: `player` (Entities.player.Player), `score`, `time_left`, `time_limit`, `phase`,
`done`, `won`, `finishing` (флаг/замок), `in_pipe` (труба), `cutscene` (любая заставка — нажатия
не действуют), `on_ground`, `progress` (0..1 до двери замка, только по основному уровню), `max_x`,
`spawn` (старт уровня), `geometry` (`pole: Flagpole(x, bottom, top)`, `castle`, `door_x`, `left`, `right`),
`warps` (все `Warp(x, top, room, exit_x, exit_top)`), `warp_under_player()` (труба-вход под ногами или None),
`rooms_visited`, `totals` (`coins`, `block_hits`, `enemies`, `secrets` — сколько всего на уровне с комнатами),
`flag_bonus`, `flag_fraction`, `time_bonus`, `death_cause` ("pit" | "enemy" | "time"),
`stats` (Counter: coin, block, enemy, stomp, secret, flag, time_bonus, hits, deaths + `*_points`), `summary()`.

События `Event(kind, points, info)`:
- `coin`; `block` (удар по «?», info: x, y, left — сколько ударов осталось); `bump` (удар в пустой «?», 0 очков);
- `enemy` (враг убит; info: type, via = "stomp" | "bump" | "other", x, y, chain), `stomp` (Koopa ушла в панцирь; очки по цепочке);
- `pipe` (info: dir = "down" | "side" | "up", room), `secret` (первый вход в комнату, info: room);
- `hit`, `flag`, `time_bonus`, `complete`, `death`.

Фазы: `playing` → (касание штока) `flagpole` → `to_castle` → `complete`. Смерть — `dead`.
Трубы: `playing` →(«вниз» на трубе-входе) `pipe_down` → комната `playing` →(вправо в устье) `pipe_side` →
уровень `pipe_up` (вылезает из трубы-выхода) → `playing`. Таймер в заставках стоит.

## Правила очков (core/rules.py)

| Что | Очки |
|---|---|
| Монета | 100 |
| Удар по блоку «?» снизу (1–4 удара в блоке) | 100 за удар |
| Враг: прыжок сверху (Goomba — убит; Koopa — в панцирь, второй прыжок — убита) | 100 сразу, цепочка без приземления: 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000 |
| Враг, стоявший на блоке, по которому ударили снизу | 100 |
| Флагшток: доля высоты захвата ≥0.95 / ≥0.70 / ≥0.45 / ≥0.20 / ниже | 5000 / 2000 / 800 / 400 / 100 |
| Вход в замок: за каждую оставшуюся целую секунду таймера | 20 |

Смерть оставляет набранные очки, но флаг и бонус за время не начисляются.
**Побеждает тот, кто набрал больше очков**, а не тот, кто быстрее прошёл.

## Физика (замерено на GameWorld)

- Тайл 32×32. Центры блоков строго на `32*i + 16` по обеим осям (иначе враги откатятся на медленную физику arcade).
- Игрок: хитбокс 26×32, скорость 4 px/кадр, гравитация 0.5, прыжок: подъём 217.5 px (6.8 тайла), 60 кадров в воздухе, 240 px (7.5 тайла) по горизонтали на одной высоте. Высота прыжка не зависит от удержания.
- Враги: 1.5 px/кадр, разворачиваются у стены, в пропасть падают. Оживают, когда игрок ближе 900 px по X.
- Ниже y = −100 — смерть. Здоровье игрока 3, после удара 1 с неуязвимости.
- Блок «?» (`ScoreBlock`) — **твёрдый**, как кирпич: на нём стоят, сквозь него не пройти, враги от него
  разворачиваются. Удар головой снизу — +100, пока в блоке есть удары (1–4); один прыжок — один удар.
  Пустой блок остаётся стеной. Чтобы блок можно было выбить, под ним должно быть место для прыжка:
  низ блока на 3–4 тайла выше пола (голова стоящего игрока на 1 тайл выше пола). Блок на пути прыжка
  через яму — **препятствие**: он обрывает прыжок. Монета не твёрдая, собирается касанием.
- Труба-вход: игрок стоит на ней (центр не дальше 20 px от середины трубы) и держит «вниз».

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

### Секретные комнаты

```json
"warps": [{
  "id": "coins1",                     // необязательно; ключ комнаты
  "pipe": [1872, 304],                // центр ЛЕВОГО ВЕРХНЕГО тайла трубы-входа в уровне
  "exit": [5872, 272],                // центр левого верхнего тайла трубы, из которой вылезти
  "room": {
    "theme": "underground",
    "objects": [ ... ],               // как у уровня: Bricks, Tubes, Coins, ScoreBlock, враги, mario = точка появления
    "exit": [432, 80]                 // центр НИЖНЕГО тайла устья боковой трубы (2 тайла высотой)
  }
}]
```

Труба — 2 тайла шириной: верх `Tubes/img.png` (левый) + `img_1.png` (правый), тело `img_2.png` + `img_3.png`.
Боковая труба выхода в комнате: устье `img_11.png` (верх) + `img_10.png` (низ), горизонтальное тело `img_8.png` / `img_9.png`,
стык с вертикальной трубой `img_4.png`…`img_7.png`. Игрок выходит, упираясь в устье справа, стоя на полу
(низ устья = уровень пола). Выход трубы-выхода в уровне — обычная труба, над ней должно быть 2+ тайла свободно.
Координаты комнаты — свои (от 0), комната — закрытая коробка: пол, стены, потолок.

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
| Игра и интерфейс | `game.py`, `main.py`, `main_menu.py`, `ui/*.py`, `records.py`, `tools/check_ui.py`, `level_editor.py` (только сохранение лишних ключей уровня) |
| Уровни | `maps/*.json` (кроме `custom_level.json`), `maps/levels.json`, `ai/levelgen.py`, `tools/level_*.py` |
| Нейросеть | `ai/*` (кроме `ai/levelgen.py`), `tools/tournament.py` |
| Ядро (только владелец) | `core/*`, `map.py`, `Entities/*`, `Objects/*` |

Нужна правка в ядре — не править, а описать в итоговом отчёте: файл, что и зачем.
`custom_level.json` (уровень 1-1 автора): объекты не трогать, можно только добавить ключ `warps`.
