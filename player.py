from __future__ import annotations

import functools
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, IntEnum
from itertools import chain, count, islice
from random import choice
from typing import Any, Callable, Iterator

from base import Board

TEST = False

# Player template for HIVE --- ALP semestral work
# Vojta Vonasek, 2023


# PUT ALL YOUR IMPLEMENTATION INTO THIS FILE

type BoardData = dict[int, dict[int, list[str]]]
type BoardDataBrute = dict[int, dict[int, str]]
type Cell = tuple[int, int]
type Direction = tuple[int, int]
# list[str, int, int, int, int] | list[str, None, None, int, int]
type MoveBrute = list[Any]

DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


def length_of_iter[T](iterator: Iterator[T]) -> int:
    """Count the number of elements in an iterator."""
    return len(tuple(iterator))


def floodfill[T](
    visited: set[Cell],
    queue: deque[Cell],
    next_fn: Callable[[Cell], Iterator[Cell]],
    map_fn: Callable[[Cell], T] = lambda x: x,
) -> Iterator[T]:
    """Run floodfill algorithm."""
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        yield map_fn(current)
        queue.extend(next_fn(current))


def parse_board(string: str) -> BoardDataBrute:
    """Parse board from string."""
    board: BoardDataBrute = {}
    lines = string.splitlines()
    for q, line in enumerate(lines):
        for p, char in enumerate(line.split(), start=-(q // 2)):
            board.setdefault(p, {})
            board[p][q] = char if char != "." else ""

    return board


def convert_board(board: BoardDataBrute) -> BoardData:
    """
    Convert board to internal representation.

    Utilizes lists instead of strings for faster manipulations.
    """
    return {p: {q: list(board[p][q]) for q in board[p]} for p in board}


class Criteria(IntEnum):
    """
    Criteria for the evaluation function.

    Used to index the evaluation tables.
    """

    BASE = 0
    """Base points for every piece on the board"""
    BLOCKING_RIVAL = 1
    """Piece (other than queen and beetle) has only one neighbor - rival"""
    BEETLE_BLOCKING = 2
    """Beetle is on top of a rival's piece"""
    BEETLE_BLOCKING_QUEEN_BONUS = 3
    """Beetle is on top of the rival's queen (adds to `BEETLE_BLOCKING`)"""
    QUEEN_NEIGHBOR = 4
    """Penalty for the every neighbor of queen"""
    QUEEN_SURROUNDED = 5
    """Penalty if the queen is completely surrounded"""
    QUEEN_BLOCKED = 6
    """Penalty if the queen can't move"""
    BLOCKING_FRIEND = 7
    """Piece (other than queen and beetle) has only one neighbor - friend"""


EVAL_TABLE_MY = [1, -500, 200, 1000, -400, -1000000, -400, 200, -50]
EVAL_TABLE_RIVAL = [-1, 600, -200, -800, 400, 1000000, 400, -100, 50]


def check_blocking(
    player: Player,
    cell: Cell,
    table: list[int],
    *,
    target_player: bool,
    only_my: bool = False,
) -> int:
    """Check if the given piece is blocking another piece."""
    neighbors = player.neighbors(cell)
    neighbor = next(neighbors, None)

    if neighbor and not next(neighbors, None):
        if not player.is_my_cell(neighbor, target_player):
            return table[Criteria.BLOCKING_RIVAL] if not only_my else 0

        return table[Criteria.BLOCKING_FRIEND]

    return 0


def check_beetle_blocking(
    pieces: list[str],
    table: list[int],
    *,
    piece_is_upper: bool,
) -> int:
    """Check if the given cell is blocked by a beetle."""
    if len(pieces) <= 1:
        return 0

    second_piece = pieces[-2]

    if second_piece.isupper() == piece_is_upper:
        return -table[Criteria.BEETLE_BLOCKING]

    score = table[Criteria.BEETLE_BLOCKING]

    if second_piece.upper() == Piece.Queen:
        score += table[Criteria.BEETLE_BLOCKING_QUEEN_BONUS]

    return score


def evaluate_cell(
    player: Player,
    cell: Cell,
    *,
    target_player: bool,
    rivals_queen: Cell | None = None,
) -> tuple[int, State]:
    """
    Evaluate the given cell from the POV of the given player.

    Uses the evaluation tables `EVAL_TABLE_MY` and `EVAL_TABLE_RIVAL`,
    following the evaluation criteria in `Criteria`
    """
    top_piece = player.top_piece_in(cell)
    piece_is_upper = top_piece.isupper()
    my = piece_is_upper == target_player
    piece = Piece.from_str(top_piece)

    table = EVAL_TABLE_MY if my else EVAL_TABLE_RIVAL

    score = table[Criteria.BASE]

    score += check_blocking(
        player,
        cell,
        table,
        target_player=target_player,
        only_my=piece in {Piece.Beetle, Piece.Queen},
    )
    if rivals_queen:
        score -= 5 * player.distance(*cell, *rivals_queen)

    if piece == Piece.Beetle:
        score += check_beetle_blocking(
            player[cell],
            table,
            piece_is_upper=piece_is_upper,
        )

    if piece == Piece.Queen:
        c = length_of_iter(player.neighbors(cell))

        if c == 6:
            return table[Criteria.QUEEN_SURROUNDED], State.LOSS if my else State.WIN

        score = table[Criteria.QUEEN_NEIGHBOR] * (c - 1)

        if player.moving_breaks_hive(cell):
            score += table[Criteria.QUEEN_BLOCKED]

    return score, State.RUNNING


evaluated = 0


def evaluate_position(player: Player, *, target_player: bool) -> tuple[int, State]:
    """Evaluate the position from the POV of the given player."""
    global evaluated

    evaluated += 1

    score = 0

    rivals_queen_str = Piece.Queen.upper() if not target_player else Piece.Queen.lower()

    rivals_queen = next(
        (cell for cell in player.hive if rivals_queen_str in player[cell]),
        None,
    )

    for cell in player.hive:
        if player.is_my_cell(cell, target_player):
            piece_score, game_state = evaluate_cell(
                player,
                cell,
                target_player=target_player,
                rivals_queen=rivals_queen,
            )
        else:
            piece_score, game_state = evaluate_cell(
                player,
                cell,
                target_player=target_player,
                rivals_queen=None,
            )
        if game_state:
            return piece_score, game_state
        score += piece_score

    return score, State.RUNNING


class Piece(str, Enum):
    """Game pieces."""

    Queen = "Q"
    Spider = "S"
    Beetle = "B"
    Ant = "A"
    Grasshopper = "G"

    @staticmethod
    def from_str(string: str) -> Piece:
        """Convert a string to a `Piece`."""
        return Piece(string.upper())

    # overrides str for little speed bonus, since all are already upper
    def upper(self) -> str:
        """Return the string representation in upper case."""
        return self.value

    def lower(self) -> str:
        """Return the string representation in lower case."""
        return self.value.lower()


@dataclass
class Move:
    """
    Holds a move, defined by a piece, start cell and end cell.

    Start is `None` when placing a new piece from reserve.
    """

    piece: Piece
    start: Cell | None
    end: Cell

    def to_brute(self, upper: bool) -> MoveBrute:
        """Convert the move to brute representation."""
        piece = self.piece_str(upper)

        if self.start is None:
            return [piece, None, None, *self.end]

        return [piece, *self.start, *self.end]

    def piece_str(self, upper: bool) -> str:
        """Return the string representation of the used piece."""
        return self.piece.upper() if upper else self.piece.lower()

    def __str__(self) -> str:
        """Return the string representation of the move."""
        return f"{self.piece_str(True)}: {self.start} -> {self.end}"


class State(IntEnum):
    """State of the game."""

    RUNNING = 0
    WIN = 1
    LOSS = 2
    DRAW = 3

    def is_end(self) -> bool:
        """
        Check if the state represents the end of the game.

        That is WIN, LOSS or DRAW.
        """
        return self > 0

    def inverse(self) -> State:  # sourcery skip: assign-if-exp, reintroduce-else
        """
        Return the inverse state.

        Swaps the WIN and LOSS states, every other state is unchanged.
        """
        if self == State.WIN:
            return State.LOSS
        if self == State.LOSS:
            return State.WIN

        return self


class Node:
    """Node in the minimax tree."""

    move: Move
    player: Player
    score: int
    children: list[Node]
    depth: int
    state: State

    def __init__(
        self,
        move: Move,
        player: Player,
    ) -> None:
        """
        Initialize the node.

        `player` is the one holding the board, not the evaluation target. That is
        specified when calling `next_depth`.
        """
        self.move = move
        self.player = player
        self.score = 0
        self.children = []
        self.depth = 0
        self.state = State.RUNNING

    def next_depth(self, player: Player, end: float, *, target_player: bool) -> bool:
        """
        Compute the next depth using the minimax algorithm.

        First call evaluates the current position, second call creates
        the children of the current node and all next calls, call
        this function recursively on the children.

        Returns True if it finished computing the next depth before
        time specified in `end`, otherwise returns False. When that happens,
        the results should be discarded, since they are only half done and
        thus not accurate.
        """
        if self.state.is_end():
            return True

        if time.perf_counter() > end:
            return False

        self.depth += 1

        with play_move(player, self.move):
            if self.depth == 1:
                self.score, self.end = evaluate_position(
                    self.player,
                    target_player=target_player,
                )
                return True

            if self.depth == 2:
                self.children = [Node(move, player) for move in player.valid_moves]

            for child in self.children:
                if not child.next_depth(
                    player,
                    end,
                    target_player=not target_player,
                ):
                    return False

            self.score, self.state = self.evaluate_children()

        return True

    def evaluate_children(self) -> tuple[int, State]:
        """Evaluate the children of the current node."""
        if not self.children:
            return 0, State.DRAW

        children = self.children

        children.sort(reverse=True)

        depth = self.depth
        if depth <= 1:
            limit = 10
        elif depth <= 3:
            limit = 5
        elif depth <= 5:
            limit = 3
        else:
            limit = 2

        self.children = children[:limit]

        best = self.children[0]

        return -best.score, best.state.inverse()

    def __gt__(self, other: Node) -> bool:
        """Compare nodes by score."""
        return self.score > other.score

    def __str__(self) -> str:
        """Return a string representation of the node."""
        if self.state.is_end():
            return f"{self.move}: {self.score} {self.state}"

        return f"{self.move}: {self.score}"


@contextmanager
def lift_piece(player: Player, cell: Cell) -> Iterator[str]:
    """Lifts a piece from the board for the duration of the context."""
    # ^ single line docstrings should fit on one line, as per PEP 257
    piece = player.remove_piece_from_board(cell)
    yield piece
    player.add_piece_to_board(cell, piece)


@contextmanager
def play_move(player: Player, move: Move) -> Iterator[None]:
    """Plays a move for the duration of the context."""
    player.play_move(move)
    yield
    player.reverse_move(move)


class Player(Board):
    """
    A player for the hive game. Public API includes the constructor and the move method.

    ## State:
    Inner state consists of a set of cells that together create the hive. This is done
    in order to speed up lot of the calculations inside and thus all of the methods
    implicitly depends on it. The properties should be independent and stateless,
    unless explicitly specified otherwise in their docstrings.
    """

    _board: BoardData
    hive: set[Cell]
    cycles: list[set[Cell]]
    __cached_cycles: dict[int, list[set[Cell]]]
    __cycles_need_update: bool

    def __init__(
        self,
        player_name: str,
        my_is_upper: bool,
        size: int,
        my_pieces: dict[str, int],
        rival_pieces: dict[str, int],
    ) -> None:
        """
        Create a new player.

        *Note: the API has to stay this way to be compatible with Brute*
        """
        super().__init__(my_is_upper, size, my_pieces, rival_pieces)
        self.playerName = player_name
        self.algorithmName = "maneren"
        self._board = convert_board(self.board)
        self.hive = set(self.nonempty_cells)
        self.cycles = []
        self.__cached_cycles = {}
        self.__cycles_need_update = True

    @property
    def upper(self) -> bool:
        """The player uses uppercased pieces."""
        return self.myColorIsUpper

    @property
    def cells(self) -> Iterator[Cell]:
        """Iterator over all cells."""
        return ((p, q) for p in self.board for q in self.board[p])

    @property
    def empty_cells(self) -> Iterator[Cell]:
        """Iterator over all empty cells."""
        return (cell for cell in self.cells if self.is_empty(cell))

    @property
    def nonempty_cells(self) -> Iterator[Cell]:
        """Iterator over all nonempty cells."""
        return (cell for cell in self.cells if not self.is_empty(cell))

    @property
    def my_pieces_on_board(self) -> Iterator[tuple[Piece, Cell]]:
        """
        Iterator over all my pieces on the board.

        Uses self.hive directly.
        """
        return (
            (Piece.from_str(self.top_piece_in(cell)), cell)
            for cell in self.hive
            if self.is_my_cell(cell)
        )

    @property
    def my_placable_pieces(self) -> Iterator[Piece]:
        """Iterator over all my placable pieces."""
        return (
            Piece.from_str(piece) for piece, count in self.myPieces.items() if count > 0
        )

    @property
    def my_movable_pieces(self) -> Iterator[tuple[Piece, Cell]]:
        """
        Iterator over all my movable pieces.

        Uses self.hive transitively.
        """
        return (
            (piece, cell)
            for piece, cell in self.my_pieces_on_board
            if not self.moving_breaks_hive(cell)
        )

    @property
    def valid_placements(self) -> Iterator[Cell]:
        """
        Iterator over all valid placements.

        Uses self.hive transitively and expects at least one piece to be already placed
        """
        return (
            cell
            for cell in self.cells_around_hive
            if self.neighbors_only_my_pieces(cell)
        )

    @property
    def valid_moves(self) -> Iterator[Move]:
        """
        Iterator over all valid moves.

        Uses self.hive transitively
        """
        mapping = {
            Piece.Ant: self.ants_moves,
            Piece.Queen: self.queens_moves,
            Piece.Beetle: self.beetles_moves,
            Piece.Grasshopper: self.grasshoppers_moves,
            Piece.Spider: self.spiders_moves,
        }

        place_iter = (
            Move(piece, None, cell)
            for cell in self.valid_placements
            for piece in self.my_placable_pieces
        )

        move_iter = (
            move
            for piece, cell in self.my_movable_pieces
            for move in mapping[piece](cell)
        )

        if self.my_piece_remaining(Piece.Queen) > 0 and self.myMove == 3:
            return map(
                functools.partial(Move, Piece.Queen, None),
                self.valid_placements,
            )

        return chain(place_iter, move_iter) if self.myMove >= 4 else place_iter

    @property
    def cells_around_hive(self) -> set[Cell]:
        """Set of all cells around the hive."""
        return {
            neighbor
            for cell in self.hive
            for neighbor in self.empty_neighboring_cells(cell)
        }

    def move(self) -> MoveBrute:
        """
        Return a best move for the current self.board.

        Has to first properly initialize the inner state - self._board and self.hive.

        Format:
            [piece, oldP, oldQ, newP, newQ] - move from (oldP, oldQ) to (newP, newQ)
            [piece, None, None, newP, newQ] - place new at (newP, newQ)
            [] - no move is possible

        *Note: the API has to stay this way to be compatible with Brute*
        """
        end = time.perf_counter() + 0.95

        self._board = convert_board(self.board)
        self.hive = set(self.nonempty_cells)
        self.__cycles_need_update = True

        if self.myMove == 0:
            if not self.hive:
                return Move(Piece.Spider, None, (3, 6)).to_brute(self.upper)

            placement = choice(list(self.cells_around_hive))
            return Move(Piece.Spider, None, placement).to_brute(self.upper)

        if TEST or not self.tournament:
            possible_moves = list(self.valid_moves)
            return choice(possible_moves).to_brute(self.upper) if possible_moves else []

        nodes = [Node(move, self) for move in self.valid_moves]

        if not nodes:
            return []

        best, depth = self.minimax(nodes, end)

        global evaluated
        print(f"Searched to depth {depth} ({evaluated} pos): {best.score}")
        evaluated = 0

        return best.move.to_brute(self.upper)

    def minimax(self, nodes: list[Node], end: float) -> tuple[Node, int]:
        """Run the minimax algorithm on the list of nodes return the best move."""
        best = nodes[0]

        depth = 0
        for depth in count():
            if time.perf_counter() > end:
                break

            for node in nodes:
                if not node.next_depth(self, end, target_player=self.upper):
                    return best, depth - 1

            if depth <= 2:
                limit = len(nodes)
            elif depth <= 5:
                limit = 5
            else:
                limit = 2

            win_moves = (node for node in nodes if node.state == State.WIN)

            win = next(win_moves, None)

            if win:
                return win, depth

            nodes.sort(reverse=True)

            nodes = list(
                islice((node for node in nodes if node.state != State.LOSS), limit),
            )

            if not nodes:
                break

            if self.my_piece_remaining(Piece.Queen) != 0:
                for node in nodes:
                    if node.move.piece == Piece.Queen:
                        node.score += 100
                        break

            best = max(nodes)

        return best, depth

    def moving_breaks_hive(self, cell: Cell) -> bool:
        """Check if moving the given piece breaks the hive into parts."""
        if len(self[cell]) > 1:
            return False

        neighbor_groups = self.neighbor_groups(cell)

        if neighbor_groups == 1:
            return False

        if self.__cycles_need_update:
            self.update_cycles()

        return not self.is_in_cycle(cell)

    def queens_moves(self, queen: Cell) -> Iterator[Move]:
        """
        Return iterator over all valid moves for the queen.

        Queen can move one step in any direction.
        """
        with lift_piece(self, queen) as piece:
            assert piece.upper() == Piece.Queen, f"{piece} at {queen} is not a Queen"
            move = functools.partial(Move, Piece.Queen, queen)
            yield from map(move, self.valid_steps(queen, can_leave_hive=True))

    def ants_moves(self, ant: Cell) -> Iterator[Move]:
        """
        Return an iterator over all valid moves for the ant.

        Ant can move any number of steps, but always has to stay right next
        to the hive.
        """
        with lift_piece(self, ant) as piece:
            assert piece.upper() == Piece.Ant, f"{piece} at {ant} is not an Ant"

            def next_cells(cell: Cell) -> Iterator[Cell]:
                return (
                    neighbor
                    for neighbor in self.empty_neighboring_cells(cell)
                    if self.can_move_to(cell, neighbor)
                )

            move = functools.partial(Move, Piece.Ant, ant)
            visited = {ant}
            queue = deque(next_cells(ant))

            yield from floodfill(visited, queue, next_cells, move)

    def beetles_moves(self, beetle: Cell) -> Iterator[Move]:
        """
        Return an iterator over all valid moves for the beetle.

        Beetle can make one step in any direction, while also being able to climb
        on top of other pieces.
        """
        with lift_piece(self, beetle) as piece:
            assert piece.upper() == Piece.Beetle, f"{piece} at {beetle} is not a Beetle"

            move = functools.partial(Move, Piece.Beetle, beetle)

            yield from map(
                move, self.valid_steps(beetle, can_crawl_over=True, can_leave_hive=True)
            )

    def grasshoppers_moves(self, grasshopper: Cell) -> Iterator[Move]:
        """
        Return an iterator over all valid moves for the grasshopper.

        Grasshopper can jump in any direction in straght line, but always has to
        jump over at least one other piece.
        """
        with lift_piece(self, grasshopper) as piece:
            assert (
                piece.upper() == Piece.Grasshopper
            ), f"{piece} at {grasshopper} is not a Grasshopper"

            move = functools.partial(Move, Piece.Grasshopper, grasshopper)

            # for each direction
            for dp, dq in DIRECTIONS:
                # start at grasshopper's position
                current_cell = grasshopper
                skipped = False

                # move in that direction until edge of board
                while True:
                    p, q = current_cell
                    current_cell = (p + dp, q + dq)

                    if not self.in_board(current_cell):
                        break

                    # if tile is empty and
                    # if something was skipped, yield move
                    # else try different direction
                    if self.is_empty(current_cell):
                        if skipped:
                            yield move(current_cell)

                        break

                    # if tile is not empty, skip the piece
                    skipped = True

    def spiders_moves(self, spider: Cell) -> Iterator[Move]:
        """
        Return an iterator over all valid moves for the spider.

        Spider can move only exactly three steps, while staying right next
        to the hive.
        """
        with lift_piece(self, spider) as piece:
            assert piece.upper() == Piece.Spider, f"{piece} at {spider} is not a Spider"

            move = functools.partial(Move, Piece.Spider, spider)
            visited: set[Cell] = set()
            stack = [spider]

            for _ in range(3):
                next_stack: list[Cell] = []

                for item in stack:
                    visited.add(item)
                    next_stack.extend(
                        neighbor
                        for neighbor in self.valid_steps(item)
                        if neighbor not in visited
                    )

                stack = next_stack
                if not stack:
                    break

            yield from map(move, stack)

    def neighboring_cells_unchecked(self, cell: Cell) -> Iterator[Cell]:
        """Return an iterator over cells neighboring (p,q), without bound checking."""
        p, q = cell
        return ((p + dp, q + dq) for dp, dq in DIRECTIONS)

    def neighboring_cells(self, cell: Cell) -> Iterator[Cell]:
        """Return an iterator over all cells neighboring (p,q)."""
        return filter(self.in_board, self.neighboring_cells_unchecked(cell))

    def empty_neighboring_cells(self, cell: Cell) -> Iterator[Cell]:
        """Return an iterator over all cells neighboring (p,q) that are empty."""
        return (
            neighbor
            for neighbor in self.neighboring_cells(cell)
            if self.is_empty(neighbor)
        )

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        """Return an iterator over all cells neighboring (p,q) that aren't empty."""
        return (
            neighbor
            for neighbor in self.neighboring_cells(cell)
            if not self.is_empty(neighbor)
        )

    def valid_steps(
        self,
        cell: Cell,
        *,
        can_crawl_over: bool = False,
        can_leave_hive: bool = False,
    ) -> Iterator[Cell]:
        """
        Return an iterator over all cells reachable from (p,q).

        For more details see `can_move_to`.
        """
        return (
            neighbor
            for neighbor in self.neighboring_cells(cell)
            if self.can_move_to(
                cell,
                neighbor,
                can_crawl_over=can_crawl_over,
                can_leave_hive=can_leave_hive,
            )
        )

    def neighbor_groups(self, cell: Cell) -> int:
        """Return the number of groups around the given cell."""
        neighbors = self.neighboring_cells_unchecked(cell)
        first = next(neighbors)

        groups = 0
        in_group = first in self.hive

        for neighbor in neighbors:
            if neighbor in self.hive:
                in_group = True
            elif in_group:
                in_group = False
                groups += 1

        if in_group and first not in self.hive:
            groups += 1

        return groups

    def update_cycles(self) -> None:
        """
        Find all cycles in the hive.

        Runs a DFS from the given cell marking the cells visited. When already
        cell is encountered second time, a cycle is detected. If the length of the cycle
        is more than 2 (trio of neighboring cells), the cycle is returned.
        """

        def collect_path(
            cell: Cell, visited_from: dict[Cell, Cell | None]
        ) -> list[Cell]:
            """Collect a path from the given cell."""
            path = [cell]

            while path[-1] in visited_from:
                new = visited_from[path[-1]]
                if not new:
                    break
                path.append(new)

            return path

        def find_cycle(start: Cell) -> set[Cell] | None:
            """Try find a cycle in the hive."""
            stack = [start]

            visited_from: dict[Cell, Cell | None] = {start: None}

            while stack:
                new_stack = []

                for cell in stack:
                    for neighbor in self.neighbors(cell):
                        if neighbor not in visited_from:
                            new_stack.append(neighbor)
                            visited_from[neighbor] = cell
                            continue

                        if visited_from[cell] == neighbor:
                            continue

                        part1 = collect_path(neighbor, visited_from)
                        part2 = collect_path(cell, visited_from)

                        last = None

                        while part1[-1] == part2[-1]:
                            part1.pop()
                            last = part2.pop()

                        if last:
                            part2.append(last)

                        result = set(part1 + part2)

                        return result if len(result) > 3 else None

                stack = new_stack

            return None

        if not self.hive or len(self.hive) <= 6:
            return

        self.__cycles_need_update = False

        hashed = hash(tuple(self.hive))

        if hashed in self.__cached_cycles:
            self.cycles = self.__cached_cycles[hashed]
            return

        self.cycles = []

        for cell in self.hive:
            if self.neighbor_groups(cell) != 2:
                continue

            if self.is_in_cycle(cell):
                continue

            cycle = find_cycle(cell)

            if cycle and cycle not in self.cycles:
                self.cycles.append(cycle)

        self.__cached_cycles[hashed] = self.cycles

    def is_in_cycle(self, cell: Cell) -> bool:
        """Check if cell is in a cycle."""
        return any(cell in cycle for cycle in self.cycles)

    def top_piece_in(self, cell: Cell) -> str:
        """Return the top piece in given cell."""
        return self[cell][-1]

    def my_piece_remaining(self, piece: Piece) -> int:
        """Return the number of my pieces of given type."""
        piece_str = piece.upper() if self.upper else piece.lower()
        return self.myPieces[piece_str]

    def is_my_cell(self, cell: Cell, target_player: bool | None = None) -> bool:
        """Check if (p,q) is a cell owned by the player."""
        piece = self[cell][-1]

        if target_player is None:
            target_player = self.upper

        return piece.isupper() == target_player or piece.islower() != target_player

    def is_empty(self, cell: Cell) -> bool:
        """Check if (p,q) is an empty cell."""
        return not self[cell]

    def in_board(self, cell: Cell) -> bool:
        """Check if (p,q) is a valid coordinate within the board."""
        p, q = cell
        return 0 <= q < self.size and -(q // 2) <= p < self.size - q // 2

    def rotate_left(self, direction: Direction) -> Direction:
        """Return direction rotated one tile to left."""
        p, q = direction
        return p + q, -p

    def rotate_right(self, direction: Direction) -> Direction:
        """Return direction rotated one tile to right."""
        p, q = direction
        return -q, p + q

    def neighbors_only_my_pieces(self, cell: Cell) -> bool:
        """Check if all neighbors of (p,q) are owned by the player."""
        return all(self.is_my_cell(neighbor) for neighbor in self.neighbors(cell))

    def has_neighbor(self, cell: Cell) -> bool:
        """Check if (p,q) has a neighbor."""
        return any(self.neighbors(cell))

    def can_move_to(
        self,
        origin: Cell,
        target: Cell,
        *,
        can_crawl_over: bool = False,
        can_leave_hive: bool = False,
    ) -> bool:
        """
        Check if a piece can move from (p,q) to (np,nq).

        When `can_crawl_over` is True, occupied cells are also considered reachable and
        blocking is ignored.
        """
        # if target is not empty and can't crawl over, can't move
        if not self.is_empty(target) and not can_crawl_over:
            return False

        p, q = origin
        np, nq = target

        direction = (np - p, nq - q)

        lp, lq = self.rotate_left(direction)
        rp, rq = self.rotate_right(direction)

        left = (p + lp, q + lq)
        right = (p + rp, q + rq)

        # line by line:
        # left is part of the hive or
        # right is part of the hive or
        # can leave the hive and the target cell neighbors hive
        # if can crawl over else
        # not at the edge of board and (
        #   left or right is part of the hive and the other one is empty
        #   or
        #   can leave the hive and the target cell neighbors hive
        # )

        return (
            (self.in_board(left) and not self.is_empty(left))
            or (self.in_board(right) and not self.is_empty(right))
            or (can_leave_hive and self.has_neighbor(target))
            if can_crawl_over
            else (self.in_board(left) and self.in_board(right))
            and (
                self.is_empty(left) != self.is_empty(right)
                or (can_leave_hive and self.has_neighbor(target))
            )
        )

    def remove_piece_from_board(self, cell: Cell) -> str:
        """Remove the top-most piece at the given cell and return it."""
        piece = self[cell].pop()

        if self.is_empty(cell):
            self.hive.remove(cell)
            self.__cycles_need_update = True

        return piece

    def add_piece_to_board(self, cell: Cell, piece: str) -> None:
        """Place the given piece at the given cell."""
        self[cell].append(piece)
        if cell not in self.hive:
            self.hive.add(cell)
            self.__cycles_need_update = True

    def play_move(self, move: Move) -> None:
        """Play the given move."""
        piece = move.piece_str(self.upper)
        start = move.start
        end = move.end

        if start is not None:
            # remove the piece from its old position
            removed = self.remove_piece_from_board(start)
            assert removed == piece

        # add the piece to its new position
        self.add_piece_to_board(end, piece)

    def reverse_move(self, move: Move) -> None:
        """Reverse the given move."""
        piece = move.piece_str(self.upper)
        start = move.start
        end = move.end

        if start is not None:
            # add the piece back to its old position
            self.add_piece_to_board(start, piece)

        # remove the piece from its new position
        removed = self.remove_piece_from_board(end)
        assert removed == piece

    def set_board(self, board: BoardDataBrute) -> None:
        """Set the board to the given board."""
        self.board = board
        self._board = convert_board(board)
        self.hive = set(self.nonempty_cells)
        self.update_cycles()

    # allows indexing the board directly using player[cell] or player[p, q]
    def __getitem__(self, cell: Cell) -> list[str]:
        """Return the list of pieces at the given cell."""
        p, q = cell
        return self._board[p][q]

    def __setitem__(self, cell: Cell, value: str | list[str]) -> None:
        """Set the list of pieces at the given cell."""
        p, q = cell
        self._board[p][q] = value if isinstance(value, list) else list(value)

    def __str__(self) -> str:
        """Return a string representation of the board."""
        lines = []

        for q in range(self.size):
            row = [
                "".join(self[p, q]) or "."
                for p in range(-self.size, self.size)
                if self.in_board((p, q))
            ]
            offset = " " if q % 2 else ""
            lines.append(offset + " ".join(row))

        return "\n".join(lines)
