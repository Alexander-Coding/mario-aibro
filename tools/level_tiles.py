"""Тайлы уровня: из чего собирается карта и как читать готовую.

Общая часть для инструментов уровней:

- `tools/level_ascii.py` — рисует уровень символами;
- `tools/level_check.py` — проверяет структуру, проходимость и «потолок» очков;
- `tools/level_build.py` — собирает ручные уровни maps/level_*.json;
- `ai/levelgen.py` — процедурный генератор для обучения нейросети.

Модуль не импортирует arcade: ASCII-рендер и проверка структуры работают
без загрузки текстур и мгновенно.

Система координат — как у редактора и GameWorld: тайл 32×32, центр блока
на (32*tx + 16, 32*ty + 16), ось Y вверх. Земля уровня 1-1 — ряды ty 4 и 5
(верх земли на y = 192), игрок и враги стоят в ряду ty 6 (центр y = 208).

Блок «?» твёрдый, как кирпич (core/world.py кладёт его в walls): на нём
стоят, сквозь него не пройти, прыжок под ним обрывается. Выбить его можно
только снизу, стоя на полу, над которым низ блока на 3–4 тайла.

Секретные комнаты («warps» в JSON уровня) собирает RoomBuilder: закрытая
коробка с полом, стенами и потолком, точка появления наверху слева,
боковая труба выхода справа на уровне пола. LevelBuilder.warp() связывает
трубу-вход уровня, комнату и трубу, из которой игрок вылезет дальше.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

# Корень репозитория: tools/level_tiles.py → ..
BASE_DIR = Path(__file__).resolve().parent.parent
MAPS_DIR = BASE_DIR / "maps"

TILE = 32
HALF = TILE // 2

# Земля «как в 1-1»: два ряда блоков, верхний — ty 5
GROUND_BOTTOM_TY = 4
GROUND_TOP_TY = 5
# Ряд, в котором стоят игрок и враги на обычной земле
STAND_TY = GROUND_TOP_TY + 1

# Скорость бега игрока, пикселей за кадр (Entities/player.py: move_speed)
RUN_SPEED = 4
FPS = 60
# Прыжок и гравитация игрока (Entities/player.py: jump_speed, gravity)
JUMP_VELOCITY = 15.0
GRAVITY_PER_FRAME = 0.5

LevelSource = Union[str, Path, dict]


def center(t: int) -> float:
    """Центр тайла t в пикселях (одинаково для X и Y)."""
    return float(t * TILE + HALF)


def tile_of(coord: float) -> int:
    """Номер тайла по координате центра."""
    return int(math.floor(float(coord) / TILE))


def on_grid(coord: float) -> bool:
    """Центр лежит строго на 32*i + 16."""
    value = float(coord) - HALF
    return abs(value / TILE - round(value / TILE)) < 1e-6


# ===============================================================
# ВИДЫ ТАЙЛОВ
# ===============================================================

# Вид тайла → (категория, ресурс, текстура) — ровно как сохраняет редактор
KINDS: dict[str, tuple[str, str, str]] = {
    # Твёрдые блоки
    "ground": ("Blocks", "Bricks", "resources/Blocks/Bricks/block_2.png"),
    "brick": ("Blocks", "Bricks", "resources/Blocks/Bricks/img_6.png"),
    "hard": ("Blocks", "Bricks", "resources/Blocks/Bricks/img_5.png"),
    "stone": ("Blocks", "Bricks", "resources/Blocks/Bricks/Block.gif"),
    # Труба: верхний ряд — «шляпка» из двух половин, ниже — тело
    "tube_top_l": ("Blocks", "Tubes", "resources/Blocks/Tubes/img.png"),
    "tube_top_r": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_1.png"),
    "tube_l": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_2.png"),
    "tube_r": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_3.png"),
    # Боковая труба выхода из комнаты: устье (2 тайла), горизонтальное
    # тело (2 тайла) и стык 2×2 с вертикальной трубой, уходящей в потолок
    "side_mouth": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_10.png"),
    "side_mouth_top": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_11.png"),
    "side_body": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_9.png"),
    "side_body_top": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_8.png"),
    "side_joint_bl": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_7.png"),
    "side_joint_br": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_6.png"),
    "side_joint_tl": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_5.png"),
    "side_joint_tr": ("Blocks", "Tubes", "resources/Blocks/Tubes/img_4.png"),
    # Объекты (не твёрдые)
    "question": ("objects", "ScoreBlock", "resources/objects/ScoreBlock/img_4.png"),
    "coin": ("objects", "Coins", "resources/objects/Coins/img.png"),
    "flag": ("objects", "Flag", "resources/objects/Flag/img_10.png"),
    # Декорации финиша
    "stick": ("decorations", "Flag stick", "resources/decorations/Flag stick/img_9.png"),
    "ball": ("decorations", "Flag stick", "resources/decorations/Flag stick/img_8.png"),
    # Сущности
    "mario": ("Entities", "mario", "resources/Entities/mario/idle/Mario.gif"),
    "goomba": ("Entities", "goomba", "resources/Entities/goomba/idle/img.png"),
    "koopa": ("Entities", "koopa", "resources/Entities/koopa/idle/img.png"),
}

# Замок: семь текстур, вид "castle:<файл>"
for _name in ("img.png", "img_1.png", "img_2.png", "img_3.png", "img_4.png", "img_5.png", "img_6.png"):
    KINDS[f"castle:{_name}"] = ("decorations", "Castle", f"resources/decorations/Castle/{_name}")

SIDE_PIPE_KINDS = {
    "side_mouth", "side_mouth_top", "side_body", "side_body_top",
    "side_joint_bl", "side_joint_br", "side_joint_tl", "side_joint_tr",
}
# Блок «?» — твёрдый: об него бьётся и игрок, и враги
SOLID_KINDS = {
    "ground", "brick", "hard", "stone", "tube_top_l", "tube_top_r", "tube_l", "tube_r",
    "block", "question",
} | SIDE_PIPE_KINDS
ENEMY_KINDS = {"goomba", "koopa"}


def classify(item: dict) -> str:
    """Вид объекта уровня по категории, ресурсу и файлу текстуры.

    Возвращает один из ключей KINDS или обобщённый вид: "block" (другой
    твёрдый блок), "castle" (часть замка), "deco" (прочая декорация),
    "object" (прочий объект), "entity" (неизвестная сущность).
    """

    category = str(item.get("category") or "").strip().lower()
    resource = str(item.get("resource") or "").strip().lower()
    texture = Path(str(item.get("texture_path") or "").replace("\\", "/")).name.lower()

    if category in ("blocks", "block", "блоки"):
        if resource == "bricks":
            return {
                "block_2.png": "ground",
                "img_5.png": "hard",
                "img_6.png": "brick",
                "block.gif": "stone",
            }.get(texture, "block")
        if resource == "tubes":
            return {
                "img.png": "tube_top_l",
                "img_1.png": "tube_top_r",
                "img_2.png": "tube_l",
                "img_3.png": "tube_r",
                "img_4.png": "side_joint_tr",
                "img_5.png": "side_joint_tl",
                "img_6.png": "side_joint_br",
                "img_7.png": "side_joint_bl",
                "img_8.png": "side_body_top",
                "img_9.png": "side_body",
                "img_10.png": "side_mouth",
                "img_11.png": "side_mouth_top",
            }.get(texture, "block")
        return "block"

    if category in ("objects", "object", "объекты"):
        if resource == "scoreblock":
            return "question"
        if resource == "coins":
            return "coin"
        if resource == "flag":
            return "flag"
        return "object"

    if category in ("decorations", "decoration", "декорации", "decorates"):
        if resource in ("flag stick", "flagstick", "flag_stick"):
            return "ball" if texture == "img_8.png" else "stick"
        if resource == "castle":
            return "castle"
        return "deco"

    if category in ("entities", "entity", "сущности"):
        if resource in ("mario", "goomba", "koopa"):
            return resource
        return "entity"

    return "deco"


def is_solid(kind: str) -> bool:
    return kind in SOLID_KINDS


def block_hits(x: float, y: float) -> int:
    """Число ударов в блоке «?» — та же формула, что в Objects/ScoreBlock.py.

    Детерминировано позицией центра блока (в комнате — в её координатах).
    """
    return random.Random(f"{float(x):.1f}:{float(y):.1f}").randint(1, 4)


# ===============================================================
# ЗАГРУЗКА
# ===============================================================


def load_level(level: LevelSource) -> dict:
    """Путь к JSON (относительно корня репозитория или maps/) или словарь."""

    if isinstance(level, dict):
        return level

    path = Path(level)
    if not path.is_absolute():
        for base in (Path.cwd(), BASE_DIR, MAPS_DIR):
            if (base / path).exists():
                path = base / path
                break
        else:
            path = BASE_DIR / path

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@dataclass
class Cell:
    """Объект уровня, приведённый к тайлу."""

    kind: str
    tx: int
    ty: int
    x: float
    y: float
    index: int  # номер в списке objects — для сообщений проверки


def level_cells(data: dict) -> list[Cell]:
    """Все объекты уровня как тайлы (с исходными координатами)."""

    cells = []
    for index, item in enumerate(data.get("objects", [])):
        if "x" not in item or "y" not in item:
            continue
        x, y = float(item["x"]), float(item["y"])
        cells.append(Cell(classify(item), tile_of(x), tile_of(y), x, y, index))
    return cells


def solid_tiles(data: dict) -> set[tuple[int, int]]:
    """Клетки с твёрдыми блоками."""
    return {(c.tx, c.ty) for c in level_cells(data) if is_solid(c.kind)}


# Оценка «тщательного» прохождения: сколько стоит выбить блок «?» (подойти,
# прыгнуть, приземлиться — ~1.2 с на удар) и взять монету в стороне от пути.
# Это оценка, не замер: врагов в ней нет, повторный удар по тому же блоку
# на деле ~0.35 с, а монеты в воздухе берутся отдельными прыжками (~1 с на
# 2–3 монеты). Ручные уровни берут лимит по большему из формулы и замера
# (tools/level_build.measured_seconds, tools.level_check --thorough)
SECONDS_PER_BLOCK_HIT = 1.2
SECONDS_PER_COIN = 0.2
# Запас лимита над тщательным прохождением: не меньше 30 %
THOROUGH_MARGIN = 1.3


def thorough_seconds(run_tiles: float, block_hits_total: int = 0, coins_total: int = 0,
                     extra_seconds: float = 0.0) -> float:
    """Время тщательного прохождения: весь путь на 4 px/кадр, все удары по
    блокам «?», монеты в стороне и extra_seconds (комнаты). Заставки труб и
    флага таймер не тратят."""

    return (run_tiles * TILE / RUN_SPEED / FPS
            + SECONDS_PER_BLOCK_HIT * block_hits_total
            + SECONDS_PER_COIN * coins_total
            + extra_seconds)


def time_limit_for(run_tiles: float, thorough: float = 0.0) -> int:
    """Лимит времени уровня: пробег от старта до двери и тщательное прохождение.

    Минимум — пробег на полной скорости (4 px/кадр); лимит — вдвое больше
    плюс 30 с запаса, но не меньше тщательного прохождения (все блоки «?»,
    монеты, комната) с запасом 30 %; округлённо вверх до 5 с. Бонус за
    время (20 за секунду) не перевешивает блок «?»: удар стоит 100 очков,
    а тратит ~1.2 с — 24 очка бонуса.
    """

    run_seconds = run_tiles * TILE / RUN_SPEED / FPS
    need = max(2.0 * run_seconds + 30.0, THOROUGH_MARGIN * thorough)
    return int(5 * math.ceil(need / 5))


# ===============================================================
# СБОРЩИК УРОВНЯ
# ===============================================================

# Замок из уровня 1-1: (смещение от штока по X, y, текстура).
# Рассчитан на землю с верхом на y = 192 (ряд ty 5).
CASTLE_LAYOUT: tuple[tuple[int, int, str], ...] = (
    (224, 208, "img_2.png"), (224, 240, "img_2.png"), (224, 272, "img.png"),
    (256, 208, "img_5.png"), (256, 240, "img_5.png"), (256, 272, "img_2.png"),
    (256, 304, "img_5.png"), (256, 336, "img.png"),
    (288, 208, "img_4.png"), (288, 240, "img_4.png"), (288, 272, "img_2.png"),
    (288, 304, "img_3.png"), (288, 336, "img.png"),
    (320, 208, "img_4.png"), (320, 240, "img_4.png"), (320, 272, "img_2.png"),
    (320, 304, "img_5.png"), (320, 336, "img.png"),
    (352, 208, "img_3.png"), (352, 240, "img_3.png"), (352, 272, "img_2.png"),
    (352, 304, "img_3.png"), (352, 336, "img.png"),
    (384, 208, "img_2.png"), (384, 240, "img_2.png"), (384, 272, "img.png"),
)

# Флагшток 1-1: 12 тайлов штока над основанием, шар сверху
POLE_STICK_TILES = 12
# Сколько тайлов земли за штоком: замок кончается на +12, земля до +13
FINISH_TAIL_TILES = 13
# Дверь замка — на +304 px от штока (9.5 тайла)
DOOR_OFFSET_PX = 304


@dataclass
class LevelBuilder:
    """Сборка уровня по тайлам: одна клетка — один объект.

    Все координаты — номера тайлов. Ряды: земля ty 4–5, над ней ty 6.
    Методы возвращают self, чтобы описание уровня читалось сверху вниз.
    """

    name: str
    description: str = ""
    cells: dict[tuple[int, int], str] = field(default_factory=dict)
    spawn: Optional[tuple[int, int]] = None
    pole_tx: Optional[int] = None
    time_limit: Optional[int] = None
    # Секретные комнаты: готовые записи "warps" (см. warp())
    warps: list[dict] = field(default_factory=list)

    # -----------------------------------------------------------
    # БАЗОВЫЕ ОПЕРАЦИИ
    # -----------------------------------------------------------

    def put(self, kind: str, tx: int, ty: int, *, replace: bool = True) -> "LevelBuilder":
        if kind not in KINDS:
            raise ValueError(f"Неизвестный вид тайла: {kind}")
        if tx < 0 or ty < 0:
            raise ValueError(f"Тайл вне уровня: {kind} ({tx}, {ty})")
        if replace or (tx, ty) not in self.cells:
            self.cells[(tx, ty)] = kind
        return self

    def clear(self, tx0: int, tx1: int, ty0: int = 0, ty1: int = 40) -> "LevelBuilder":
        """Убрать всё в прямоугольнике (границы включительно)."""
        for key in [k for k in self.cells if tx0 <= k[0] <= tx1 and ty0 <= k[1] <= ty1]:
            del self.cells[key]
        return self

    def kind_at(self, tx: int, ty: int) -> Optional[str]:
        return self.cells.get((tx, ty))

    def solid_at(self, tx: int, ty: int) -> bool:
        kind = self.cells.get((tx, ty))
        return kind is not None and is_solid(kind)

    def surface(self, tx: int) -> Optional[int]:
        """Ряд, в котором стоит игрок над столбцом tx.

        Считается от низа: берём самый нижний твёрдый блок столбца (землю
        или платформу над ямой) и поднимаемся, пока над ним твёрдое, —
        труба, лестница, стол. Висящие выше ряды кирпичей и блоков «?»
        опорой не считаются: враг и старт встают на землю под ними.
        None — в столбце нет опоры (яма).
        """
        rows = {ty for (x, ty), kind in self.cells.items() if x == tx and is_solid(kind)}
        if not rows:
            return None
        ty = min(rows)
        while ty + 1 in rows:
            ty += 1
        return ty + 1

    def floor_below(self, tx: int, ty: int) -> Optional[int]:
        """Ряд, где стоит игрок под клеткой (tx, ty): верхний твёрдый ниже + 1."""
        rows = [r for (x, r), kind in self.cells.items() if x == tx and r < ty and is_solid(kind)]
        return max(rows) + 1 if rows else None

    # -----------------------------------------------------------
    # РЕЛЬЕФ
    # -----------------------------------------------------------

    def ground(self, tx0: int, tx1: int, top: int = GROUND_TOP_TY) -> "LevelBuilder":
        """Земля от tx0 до tx1 включительно; верх — ряд top (обычно 5)."""
        for tx in range(tx0, tx1 + 1):
            for ty in range(GROUND_BOTTOM_TY, top + 1):
                self.put("ground", tx, ty)
        return self

    def pit(self, tx0: int, tx1: int) -> "LevelBuilder":
        """Яма: убрать всё твёрдое в столбцах tx0..tx1 ниже ряда 6."""
        for key in [k for k, kind in self.cells.items()
                    if tx0 <= k[0] <= tx1 and k[1] < STAND_TY and is_solid(kind)]:
            del self.cells[key]
        return self

    def column(self, tx: int, ty0: int, ty1: int, kind: str = "hard") -> "LevelBuilder":
        for ty in range(ty0, ty1 + 1):
            self.put(kind, tx, ty)
        return self

    def row(self, tx0: int, tx1: int, ty: int, kind: str = "brick") -> "LevelBuilder":
        for tx in range(tx0, tx1 + 1):
            self.put(kind, tx, ty)
        return self

    def tube(self, tx: int, height: int, base: int = STAND_TY) -> "LevelBuilder":
        """Труба шириной 2 (tx, tx+1) и высотой height тайлов от ряда base."""
        if height < 1:
            raise ValueError("Труба ниже одного тайла")
        top = base + height - 1
        for ty in range(base, top):
            self.put("tube_l", tx, ty)
            self.put("tube_r", tx + 1, ty)
        self.put("tube_top_l", tx, top)
        self.put("tube_top_r", tx + 1, top)
        return self

    def stairs_up(self, tx: int, steps: int, base: int = STAND_TY, width: int = 1,
                  kind: str = "hard") -> "LevelBuilder":
        """Лестница вверх: ступень i (0..steps-1) высотой i+1, шириной width."""
        for i in range(steps):
            for w in range(width):
                self.column(tx + i * width + w, base, base + i, kind)
        return self

    def stairs_down(self, tx: int, steps: int, base: int = STAND_TY, width: int = 1,
                    kind: str = "hard") -> "LevelBuilder":
        """Лестница вниз: первая ступень высотой steps, последняя — 1."""
        for i in range(steps):
            for w in range(width):
                self.column(tx + i * width + w, base, base + steps - 1 - i, kind)
        return self

    # -----------------------------------------------------------
    # ОЧКИ И ВРАГИ
    # -----------------------------------------------------------

    def coin(self, tx: int, ty: int) -> "LevelBuilder":
        return self.put("coin", tx, ty, replace=False)

    def coins(self, tx0: int, tx1: int, ty: int) -> "LevelBuilder":
        for tx in range(tx0, tx1 + 1):
            self.coin(tx, ty)
        return self

    def question(self, tx: int, ty: int) -> "LevelBuilder":
        return self.put("question", tx, ty)

    def blocks(self, tx: int, ty: int, pattern: str) -> "LevelBuilder":
        """Ряд блоков по шаблону слева направо: «?» — блок «?», «B» —
        кирпич, «X» — камень, «o» — монета, «.» — пропуск.

        Блок «?» твёрдый: ряд над землёй ставят на 3–4 тайла выше пола
        (ty = STAND_TY + 3 или + 4), не над ямой и не над разбегом прыжка.
        """
        kinds = {"?": "question", "B": "brick", "X": "hard", "o": "coin"}
        for i, char in enumerate(pattern):
            if char in kinds:
                self.put(kinds[char], tx + i, ty, replace=char != "o")
        return self

    def enemy_pen(self, tx0: int, tx1: int, ty: int, kind: str = "goomba",
                  at: Optional[Iterable[int]] = None) -> "LevelBuilder":
        """Враги на ряду блоков tx0..tx1 (ряд ty — сами блоки), с бортиками
        из кирпича по краям, чтобы не сошли вниз до прихода игрока.

        Бортики — в ряду ty+1 над крайними блоками; враги ходят между ними.
        Удар снизу по блоку под врагом убивает его (+100), как в SMB.
        """
        self.put("brick", tx0, ty + 1)
        self.put("brick", tx1, ty + 1)
        for tx in (at if at is not None else (tx0 + 1 + (tx1 - tx0 - 2) // 2,)):
            self.put(kind, tx, ty + 1)
        return self

    def coin_arc(self, tx0: int, tx1: int, base: int, peak: int) -> "LevelBuilder":
        """Монеты по параболе от (tx0, base) до (tx1, base) с вершиной base+peak.

        Дорожка повторяет траекторию прыжка и подсказывает её.
        """
        span = max(tx1 - tx0, 1)
        for tx in range(tx0, tx1 + 1):
            u = (tx - tx0) / span
            ty = base + int(round(peak * 4 * u * (1 - u)))
            if not self.solid_at(tx, ty):
                self.coin(tx, ty)
        return self

    def jump_trail(self, tx: int, ty: Optional[int] = None, *, speed: float = 4.0,
                   direction: int = 1, every: int = 1, skip: int = 1,
                   land_ty: Optional[int] = None,
                   until_col: Optional[int] = None) -> "LevelBuilder":
        """Дорожка монет по настоящей траектории прыжка.

        Прыжок из центра тайла (tx, ty) — ty это ряд, где стоит игрок, —
        со скоростью speed px/кадр. Монета ставится в тайл, где оказался
        центр игрока, в каждой every-й новой колонке; первые skip колонок
        пропускаются (над точкой отрыва монета не нужна). Дорожка кончается,
        когда игрок опустился до ряда land_ty (по умолчанию — ряда старта).
        Прыжок с разбега из этой точки собирает все монеты дорожки.
        """
        if ty is None:
            ty = self.surface(tx) or STAND_TY
        land = ty if land_ty is None else land_ty
        x = center(tx)
        bottom = float(ty * TILE)
        vy = JUMP_VELOCITY
        seen: set[int] = set()
        for _ in range(400):
            vy -= GRAVITY_PER_FRAME
            bottom += vy
            x += speed * direction
            if vy < 0 and bottom < land * TILE:
                break
            col = tile_of(x)
            if until_col is not None and col * direction > until_col * direction:
                break
            if col in seen:
                continue
            seen.add(col)
            offset = abs(col - tx)
            if offset < skip or (offset - skip) % every:
                continue
            row = tile_of(bottom + HALF)
            if not self.solid_at(col, row) and (col, row) not in self.cells:
                self.coin(col, row)
        return self

    def enemy(self, kind: str, tx: int, ty: Optional[int] = None) -> "LevelBuilder":
        """Враг на поверхности столбца (или в заданном ряду)."""
        if kind not in ENEMY_KINDS:
            raise ValueError(f"Неизвестный враг: {kind}")
        if ty is None:
            ty = self.surface(tx)
            if ty is None:
                raise ValueError(f"Враг над ямой: {kind} в столбце {tx}")
        return self.put(kind, tx, ty)

    def start(self, tx: int, ty: Optional[int] = None) -> "LevelBuilder":
        if ty is None:
            ty = self.surface(tx) or STAND_TY
        self.spawn = (tx, ty)
        return self

    # -----------------------------------------------------------
    # СЕКРЕТНЫЕ КОМНАТЫ
    # -----------------------------------------------------------

    def pipe_top(self, tx: int) -> int:
        """Ряд верхнего тайла трубы, левая половина которой в столбце tx."""
        rows = [ty for (x, ty), kind in self.cells.items() if x == tx and kind == "tube_top_l"]
        if not rows:
            raise ValueError(f"В столбце {tx} нет трубы")
        return max(rows)

    def warp(self, pipe_tx: int, exit_tx: int, room: "RoomBuilder",
             key: Optional[str] = None) -> "LevelBuilder":
        """Труба-вход (левый столбец pipe_tx) → комната → труба-выход exit_tx.

        Обе трубы уже поставлены tube(). Над трубой-выходом должно быть
        2+ тайла свободно, за ней — земля без врагов рядом (проверяет
        tools/level_check.py и прогоном в GameWorld).
        """
        self.warps.append(warp_entry(
            (pipe_tx, self.pipe_top(pipe_tx)), (exit_tx, self.pipe_top(exit_tx)), room,
            key or f"room{len(self.warps) + 1}",
        ))
        return self

    # -----------------------------------------------------------
    # ФИНИШ
    # -----------------------------------------------------------

    def finish(self, pole_tx: int) -> "LevelBuilder":
        """Флагшток и замок как в 1-1: основание, шток, шар, флаг, замок.

        Земля под штоком и до конца замка — ровная, без ям и препятствий.
        """
        self.pole_tx = pole_tx

        # Ровная земля от штока до конца замка, над ней пусто
        self.clear(pole_tx, pole_tx + FINISH_TAIL_TILES, STAND_TY, 40)
        self.ground(pole_tx - 1, pole_tx + FINISH_TAIL_TILES)

        # Основание, шток, шар, флаг
        self.put("hard", pole_tx, STAND_TY)
        for i in range(POLE_STICK_TILES):
            self.put("stick", pole_tx, STAND_TY + 1 + i)
        self.put("ball", pole_tx, STAND_TY + 1 + POLE_STICK_TILES)
        self.put("flag", pole_tx - 1, STAND_TY + POLE_STICK_TILES)

        # Замок: смещения в пикселях от штока, как в 1-1
        for dx, y, texture in CASTLE_LAYOUT:
            self.put(f"castle:{texture}", pole_tx + dx // TILE, tile_of(y))
        return self

    def final_stairs(self, tx: int, height: int = 8, top_width: int = 1,
                     run: int = 4) -> int:
        """Лестница к флагу, как в 1-1, и финиш за ней. Возвращает столбец штока.

        Ступени высотой 1..height начиная со столбца tx, наверху площадка
        шириной top_width, дальше run тайлов ровной земли и флагшток.
        С площадки высотой 8 и run = 4 прыжок с разбега хватает шток за
        самый верх (5000): игрок долетает до штока через ~26 кадров, на
        подъёме, в окне 15–44 кадра, когда он выше 95 % штока.
        """
        self.stairs_up(tx, height)
        last = tx + height - 1
        for w in range(1, top_width):
            self.column(last + w, STAND_TY, STAND_TY + height - 1)
        last += top_width - 1
        pole = last + run + 1
        self.finish(pole)
        return pole

    # -----------------------------------------------------------
    # ВЫГРУЗКА
    # -----------------------------------------------------------

    @property
    def width_tiles(self) -> int:
        return max(tx for tx, _ in self.cells) + 1 if self.cells else 0

    def run_tiles(self) -> float:
        """Путь от старта до двери замка в тайлах."""
        if self.spawn is None or self.pole_tx is None:
            return float(self.width_tiles)
        door_x = self.pole_tx * TILE + HALF + DOOR_OFFSET_PX
        return (door_x - center(self.spawn[0])) / TILE

    def block_hits_total(self) -> int:
        return sum(block_hits(center(tx), center(ty))
                   for (tx, ty), kind in self.cells.items() if kind == "question")

    def coins_total(self) -> int:
        return sum(kind == "coin" for kind in self.cells.values())

    def thorough_seconds(self) -> float:
        """Тщательное прохождение: весь путь, все удары, монеты, комнаты."""
        rooms = sum(float(w.get("_seconds", 0.0)) for w in self.warps)
        return thorough_seconds(self.run_tiles(), self.block_hits_total(),
                                self.coins_total(), rooms)

    def objects(self) -> list[dict]:
        """Объекты в формате редактора, старт — сущностью mario."""
        if self.spawn is None:
            raise ValueError("Нет точки старта")

        objects = []
        cells = dict(self.cells)
        cells[self.spawn] = "mario"

        # Порядок: блоки, объекты, декорации, сущности — по X, затем по Y
        order = {"Blocks": 0, "objects": 1, "decorations": 2, "Entities": 3}
        for (tx, ty), kind in sorted(
            cells.items(), key=lambda kv: (order[KINDS[kv[1]][0]], kv[0][0], kv[0][1])
        ):
            category, resource, texture = KINDS[kind]
            objects.append({
                "x": center(tx),
                "y": center(ty),
                "category": category,
                "resource": resource,
                "texture_path": texture,
            })
        return objects

    def to_level(self) -> dict:
        """Словарь уровня в формате редактора (+ name, description, time_limit, warps)."""
        if self.spawn is None:
            raise ValueError("У уровня нет точки старта")

        # По умолчанию — лимит по пробегу (ai/levelgen.py добавляет к нему
        # своё время на сбор); ручные уровни (tools/level_build.py) ставят
        # time_limit сами — по тщательному прохождению, см. thorough_seconds()
        time_limit = self.time_limit or time_limit_for(self.run_tiles())
        level = {
            "name": self.name,
            "description": self.description,
            "time_limit": time_limit,
            "objects": self.objects(),
        }
        if self.warps:
            level["warps"] = [{k: v for k, v in w.items() if not k.startswith("_")}
                              for w in self.warps]
        return level


# ===============================================================
# СЕКРЕТНАЯ КОМНАТА
# ===============================================================

# Ряды комнаты: пол — ty 0–1, игрок стоит в ряду 2
ROOM_FLOOR_TY = 1
ROOM_STAND_TY = ROOM_FLOOR_TY + 1


@dataclass
class RoomBuilder(LevelBuilder):
    """Секретная комната: закрытая коробка в своих координатах (от 0).

    box() ставит пол (ряды 0–1), стены и потолок; side_exit() — боковую
    трубу выхода у правой стены: устье на уровне пола, за ним тело и стык
    с вертикальной трубой до потолка. Точка появления — наверху слева:
    игрок падает в комнату, как в Super Mario Bros.

        room = RoomBuilder("монеты").box(20, 12).start(2, 9)
        room.coins(4, 10, 5)
        room.side_exit()
    """

    theme: str = "underground"
    width: int = 0
    height: int = 0
    exit_tile: Optional[tuple[int, int]] = None

    def box(self, width: int, height: int = 12, wall: str = "brick",
            floor: str = "ground") -> "RoomBuilder":
        """Коробка width×height тайлов: пол в два ряда, стены, потолок."""
        self.width, self.height = width, height
        for tx in range(width):
            for ty in range(ROOM_FLOOR_TY + 1):
                self.put(floor, tx, ty)
            self.put(wall, tx, height - 1)
        for ty in range(ROOM_STAND_TY, height - 1):
            self.put(wall, 0, ty)
            self.put(wall, width - 1, ty)
        return self

    def side_exit(self) -> "RoomBuilder":
        """Боковая труба у правой стены: устье (2 тайла) на полу, тело,
        стык 2×2 и вертикальная труба до потолка. Выход — упереться в устье."""
        mouth = self.width - 5
        base = ROOM_STAND_TY
        self.put("side_mouth", mouth, base)
        self.put("side_mouth_top", mouth, base + 1)
        self.put("side_body", mouth + 1, base)
        self.put("side_body_top", mouth + 1, base + 1)
        self.put("side_joint_bl", mouth + 2, base)
        self.put("side_joint_br", mouth + 3, base)
        self.put("side_joint_tl", mouth + 2, base + 1)
        self.put("side_joint_tr", mouth + 3, base + 1)
        for ty in range(base + 2, self.height - 1):
            self.put("tube_l", mouth + 2, ty)
            self.put("tube_r", mouth + 3, ty)
        self.exit_tile = (mouth, base)
        return self

    def run_tiles(self) -> float:
        """Путь по комнате: от точки появления до устья."""
        if self.spawn is None or self.exit_tile is None:
            return float(self.width)
        return float(abs(self.exit_tile[0] - self.spawn[0]))

    def seconds(self) -> float:
        """Время тщательного прохода комнаты (таймер в трубах стоит)."""
        return thorough_seconds(self.run_tiles(), self.block_hits_total(),
                                self.coins_total(), extra_seconds=1.0)

    def to_room(self) -> dict:
        if self.exit_tile is None:
            raise ValueError("В комнате нет выхода: side_exit()")
        return {
            "theme": self.theme,
            "objects": self.objects(),
            "exit": [center(self.exit_tile[0]), center(self.exit_tile[1])],
        }


def warp_entry(pipe: tuple[int, int], exit_pipe: tuple[int, int], room: RoomBuilder,
               key: str) -> dict:
    """Запись "warps" уровня: pipe и exit — (столбец, ряд) левого верхнего
    тайла труб. Поле _seconds (оценка времени комнаты) в JSON не пишется."""
    return {
        "id": key,
        "pipe": [center(pipe[0]), center(pipe[1])],
        "exit": [center(exit_pipe[0]), center(exit_pipe[1])],
        "room": room.to_room(),
        "_seconds": round(room.seconds(), 1),
    }


def iter_kinds(data: dict, kinds: Iterable[str]) -> list[Cell]:
    wanted = set(kinds)
    return [c for c in level_cells(data) if c.kind in wanted]
