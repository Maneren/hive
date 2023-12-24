from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Callable, Iterator, TypeVar

from base import Board

# Player template for HIVE --- ALP semestral work
# Vojta Vonasek, 2023


# PUT ALL YOUR IMPLEMENTATION INTO THIS FILE

BoardData = dict[int, dict[int, str]]
Cell = tuple[int, int]
MoveBrute = list[str, int, int, int, int] | list[str, None, None, int, int]

DIRECTIONS = ((0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0))

SQUEEZE_DIRECTION_LR_MAP = {
    (0, -1): ((-1, 0), (1, -1)),
    (0, 1): ((1, 0), (-1, 1)),
    (1, 0): ((1, -1), (0, 1)),
    (-1, 0): ((-1, 1), (0, -1)),
    (1, -1): ((1, 0), (0, -1)),
    (-1, 1): ((0, 1), (-1, 0)),
}


T = TypeVar("T")


def consume(iterator: Iterator[T]) -> None:
    import collections

    collections.deque(iterator, maxlen=0)


def floodfill(
    visited: set[Cell],
    stack: list[Cell],
    next_fn: Callable[[int, int], Iterator[Cell]],
    map_fn: Callable[[Cell], T],
) -> Iterator[T]:
    while stack:
        current = stack.pop()
        visited.add(current)
        yield map_fn(current)
        stack.extend(next_fn(*current))


def floodfill_except_first(
    visited: set[Cell],
    stack: list[Cell],
    next_fn: Callable[[int, int], Iterator[Cell]],
    map_fn: Callable[[Cell], T],
) -> Iterator[T]:
    iterator = floodfill(visited, stack, next_fn, map_fn)
    next(iterator)  # skip first element
    return iterator


def parse_board(string: str) -> BoardData:
    """
    Parse board from string
    """
    board: BoardData = {}
    lines = string.splitlines()
    for q, line in enumerate(lines):
        p = -(q // 2)

        for char in line.split():
            if p not in board:
                board[p] = {}

            board[p][q] = char if char != "." else ""
            p += 1

    return board


def cells_are_neighbors(cell1: Cell, cell2: Cell) -> bool:
    """
    Check if two cells are neighbors
    """
    p1, q1 = cell1
    p2, q2 = cell2
    return (p1 - p2, q1 - q2) in DIRECTIONS


class Piece(StrEnum):
    """
    Possible pieces
    """

    Queen = "Q"  # bee
    Spider = "S"  # spider
    Beetle = "B"  # beetle
    Ant = "A"  # ant
    Grasshopper = "G"  # grasshopper


@dataclass
class Move:
    _piece: Piece
    start: Cell | None
    end: Cell

    def to_brute(self, upper: bool) -> MoveBrute:
        piece = self._piece.value if upper else self._piece.lower()

        if self.start is None:
            return [piece, None, None, *self.end]

        return [piece, *self.start, *self.end]

    def piece_str(self, upper: bool) -> str:
        return self._piece.value if upper else self._piece.lower()

    def __str__(self) -> str:
        return f"{self.piece_str(True)}: {self.start} -> {self.end}"


class State(IntEnum):
    """
    State of the game
    """

    RUNNING = 0
    WIN = 1
    LOSS = 2
    DRAW = 3

    def is_end(self) -> bool:
        return self > 0


class Node:
    move: Move
    player_is_upper: bool
    score: int
    children: list[Node]
    depth: int
    state: State

    def __init__(
        self,
        move: Move,
        player_is_upper: bool,
        initial_score: int = 0,
    ) -> None:
        self.move = move
        self.player_is_upper = player_is_upper
        self.score = initial_score
        self.children = []
        self.depth = 0
        self.state = Node.State.RUNNING
        self.state = State.RUNNING

    def next_depth(self, player: Player) -> None:
        """
        Compute the next depth using the minimax algorithm
        """
        assert not self.state.is_end()

        self.depth += 1

        if self.depth == 1:
            self.initialize_children(player)
            return

        player.play_move(self.move)

        for child in self.children:
            child.next_depth(player)

        player.reverse_move(self.move)

    def initialize_children(self, player: Player) -> None:
        """
        Initialize the children of the current node
        """
        player.play_move(self.move)

        for move in player.valid_moves:
            self.children.append(Node(move, not self.player_is_upper))

        player.reverse_move(self.move)


class Player(Board):
    hive: set[Cell]

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
        self.hive = set(self.nonempty_cells)

    @property
    def upper(self) -> bool:
        return self.myColorIsUpper

    @property
    def cells(self) -> Iterator[Cell]:
        """
        Iterator over all cells
        """
        return ((p, q) for p in self.board for q in self.board[p])

    @property
    def empty_cells(self) -> Iterator[Cell]:
        """
        Iterator over all empty cells
        """
        return (cell for cell in self.cells if self.is_empty(*cell))

    @property
    def nonempty_cells(self) -> Iterator[Cell]:
        """
        Iterator over all nonempty cells
        """
        return (cell for cell in self.cells if not self.is_empty(*cell))

    @property
    def my_pieces(self) -> Iterator[tuple[str, Cell]]:
        """
        Iterator over all my pieces on the board. Uses self.hive
        """
        return ((self[p, q], (p, q)) for p, q in self.hive if self.is_my_cell(p, q))

    @property
    def valid_moves(self) -> Iterator[Move]:
        """
        Iterator over all valid moves
        """

        raise NotImplementedError

    @property
    def cells_around_hive(self) -> Iterator[Cell]:
        """
        Iterator over all cells around the hive
        """
        for p, q in self.hive:
            yield from self.empty_neighboring_cells(p, q)

    def move(self) -> MoveBrute:
        """
        Returns a best move for the current self.board.

        Format:
            [piece, oldP, oldQ, newP, newQ] - move from (oldP, oldQ) to (newP, newQ)
            [piece, None, None, newP, newQ] - place new at (newP, newQ)
            [] - no move is possible

        *Note: the API has to stay this way to be compatible with Brute*
        """

        self.hive = set(self.nonempty_cells)

        return []

    def hive_stays_contiguous(self, move: Move) -> bool:
        """
        Check if the hive is contiguous
        """

        self.play_move(move)

        def next_fn(p: int, q: int) -> Iterator[Cell]:
            return (
                neighbor for neighbor in self.neighbors(p, q) if neighbor not in visited
            )

        stack = [move.end]
        visited = {move.end}

        consume(floodfill(visited, stack, next_fn, lambda _: None))

        ok = len(visited) == len(self.hive)

        self.reverse_move(move)

        return ok

    def neighboring_cells(self, p: int, q: int) -> Iterator[Cell]:
        """
        Iterator over all cells neighboring the cells (p,q)
        in the hexagonal board
        """
        return (
            (p + dp, q + dq) for dp, dq in DIRECTIONS if self.in_board(p + dp, q + dq)
        )

    def empty_neighboring_cells(self, p: int, q: int) -> Iterator[Cell]:
        """
        Iterator over all cells neighboring (p,q) that are empty
        """
        return (cell for cell in self.neighboring_cells(p, q) if self.is_empty(*cell))

    def neighbors(self, p: int, q: int) -> Iterator[Cell]:
        """
        Iterator over all cells neighboring (p,q) that aren't empty
        """
        return (
            cell for cell in self.neighboring_cells(p, q) if not self.is_empty(*cell)
        )

    def has_neighbor(self, p: int, q: int) -> bool:
        """
        Check if the given cell has at least one neighbor
        """
        return any(self.neighbors(p, q))

    def horizontal(self, p: int, q: int) -> Iterator[Cell]:
        """
        Iterator over all cells on the same horizontal line as (p,q)
        """
        return ((p + dp, q) for dp in range(-q // 2, self.size - q // 2))

    def diagonal_l(self, p: int, _q: int) -> Iterator[Cell]:
        """
        Iterator over all cells on the same left diagoal line as (p,q)
        """
        return ((p, nq) for nq in range(self.size) if self.in_board(p, nq))

    def diagonal_r(self, p: int, q: int) -> Iterator[Cell]:
        """
        Iterator over all cells on the same right diagonal line as (p,q)
        """
        base_p = p + q
        return ((base_p - nq, nq) for nq in range(self.size) if self.in_board(nq, q))

    def is_my_cell(self, p: int, q: int) -> bool:
        cell = self[p, q]
        is_upper = self.myColorIsUpper

        return cell.isupper() == is_upper or cell.islower() != is_upper

    def is_empty(self, p: int, q: int) -> bool:
        return self.board[p][q] == ""

    def in_board(self, p: int, q: int) -> bool:
        """
        Check if (p,q) is a valid coordinate within the board
        """
        return 0 <= q < self.size and -(q // 2) <= p < self.size - q // 2

    def is_valid_move(self, piece: str, x: int, y: int) -> bool:
        return self.board[x][y][-1] == piece

    def is_valid_placement(self, p: int, q: int) -> bool:
        """
        Check if (p,q) is a valid placement for a new piece. Assumes
        there are already other pieces on the board
        """
        return all(self.is_my_cell(*cell) for cell in self.neighbors(p, q))

    def can_squeeze_through(self, p: int, q: int, np: int, nq: int) -> bool:
        """
        Check if a piece can move from (p,q) to (np,nq), ie. there are no other pieces
        on the sides blocking it.
        """
        direction = (np - p, nq - q)

        (lp, lq), (rp, rq) = SQUEEZE_DIRECTION_LR_MAP[direction]

        left = (p + lp, q + lq)
        right = (p + rp, q + rq)

        return any(
            self.in_board(*cell) and self.is_empty(*cell) for cell in (left, right)
        )

    def play_move(self, move: Move) -> None:
        """
        Play the given move
        """
        piece = move.piece_str(self.upper)
        start = move.start
        end = move.end

        # add the piece to its new position
        self[end] += piece

        if start is not None:
            # remove the piece from its old position
            assert self[start][-1] == piece
            self[start] = self[start][:-1]

    def reverse_move(self, move: Move) -> None:
        """
        Reverse the given move
        """
        piece = move.piece_str(self.upper)
        start = move.start
        end = move.end

        if start is not None:
            # add the piece back to its old position
            self[start] += self[end][-1]

        # remove the piece from its new position
        assert self[end][-1] == piece
        self[end] = self[end][:-1]

    ## allows indexing the board directly using player[cell] or player[p, q]
    def __getitem__(self, cell: Cell) -> str:
        p, q = cell
        return self.board[p][q]

    def __setitem__(self, cell: Cell, value: str) -> None:
        p, q = cell
        self.board[p][q] = value

    def __str__(self) -> str:
        lines = []

        for q in range(self.size):
            row = [
                self[p, q] or "."
                for p in range(-self.size, self.size)
                if self.in_board(p, q)
            ]
            offset = " " if q % 2 else ""
            lines.append(offset + " ".join(row))

        return "\n".join(lines)


def update_players(
    move: MoveBrute,
    active_player: Player,
    passive_player: Player,
) -> None:
    """write move made by activePlayer player
    this method assumes that all moves are correct, no checking is made
    """
    if not move:
        return

    animal, p, q, newp, newq = move
    if p is None and q is None:
        # placing new animal
        active_player.myPieces[animal] -= 1
        passive_player.rivalPieces = active_player.myPieces.copy()
    else:
        # just moving animal
        # delete its old position
        active_player.board[p][q] = active_player.board[p][q][:-1]
        passive_player.board[p][q] = passive_player.board[p][q][:-1]

    active_player.board[newp][newq] += animal
    passive_player.board[newp][newq] += animal


def main() -> None:
    board_size = 13
    small_figures = {
        "q": 1,
        "a": 2,
        "b": 2,
        "s": 2,
        "g": 2,
    }  # key is animal, value is how many is available for placing
    big_figures = {
        figure.upper(): val for figure, val in small_figures.items()
    }  # same, but with upper case

    p1 = Player("player1", True, board_size, small_figures, big_figures)
    p2 = Player("player2", True, board_size, big_figures, small_figures)

    filename = "output/begin.png"
    p1.saveImage(filename)

    move_idx = 0
    while True:
        move = p1.move()
        print("P1 returned", move)
        update_players(move, p1, p2)  # update P1 and P2 according to the move
        p1.saveImage("output/move-{move_idx:03d}-player1.png")

        move = p2.move()
        print("P2 returned", move)
        update_players(move, p2, p1)  # update P2 and P1 according to the move
        p1.saveImage("output/move-{move_idx:03d}-player2.png")

        move_idx += 1
        p1.myMove = move_idx
        p2.myMove = move_idx

        if move_idx > 50:
            print("End of the test game")
            break


if __name__ == "__main__":
    main()
