from __future__ import annotations

import functools
from collections import deque
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from itertools import chain
from typing import Callable, Iterator, TypeVar

from base import Board

# Player template for HIVE --- ALP semestral work
# Vojta Vonasek, 2023


# PUT ALL YOUR IMPLEMENTATION INTO THIS FILE

BoardData = dict[int, dict[int, str]]
Cell = tuple[int, int]
Direction = tuple[int, int]
MoveBrute = list[str, int, int, int, int] | list[str, None, None, int, int]

DIRECTIONS = ((0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0))

T = TypeVar("T")


def consume(iterator: Iterator[T]) -> None:
    """
    Consume the whole iterator
    """

    list(iterator)


def floodfill(
    visited: set[Cell],
    queue: deque[Cell],
    next_fn: Callable[[Cell], Iterator[Cell]],
    map_fn: Callable[[Cell], T],
) -> Iterator[T]:
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        yield map_fn(current)
        queue.extend(next_fn(current))


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

    @staticmethod
    def from_str(string: str) -> Piece:
        return Piece(string.upper())

    def upper(self) -> str:
        return self.value


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
        return self._piece.upper() if upper else self._piece.lower()

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


class LiftPiece:
    player: Player
    cell: Cell

    piece: str

    def __init__(self, player: Player, cell: Cell) -> None:
        self.player = player
        self.cell = cell

    def __enter__(self) -> str:
        self.piece = self.player.remove_piece_from_board(self.cell)
        return self.piece

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.player.add_piece_to_board(self.cell, self.piece)


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
        return (cell for cell in self.cells if self.is_empty(cell))

    @property
    def nonempty_cells(self) -> Iterator[Cell]:
        """
        Iterator over all nonempty cells
        """
        return (cell for cell in self.cells if not self.is_empty(cell))

    @property
    def my_pieces_on_board(self) -> Iterator[tuple[Piece, Cell]]:
        """
        Iterator over all my pieces on the board. Uses self.hive
        """
        return (
            (Piece.from_str(self.top_piece_in(cell)), cell)
            for cell in self.hive
            if self.is_my_cell(cell)
        )

    @property
    def my_placable_pieces(self) -> Iterator[Piece]:
        """
        Iterator over all my placable pieces
        """
        return (
            Piece.from_str(piece) for piece, count in self.myPieces.items() if count > 0
        )

    @property
    def my_movable_pieces(self) -> Iterator[tuple[Piece, Cell]]:
        """
        Iterator over all my movable pieces
        """
        return (
            (piece, cell)
            for piece, cell in self.my_pieces_on_board
            if self.moving_doesnt_break_hive(cell)
        )

    @property
    def valid_placements(self) -> Iterator[Cell]:
        """
        Iterator over all valid placements
        """
        return (
            cell
            for cell in self.cells_around_hive
            if self.neighbors_only_my_pieces(cell)
        )

    @property
    def valid_moves(self) -> Iterator[Move]:
        """
        Iterator over all valid moves
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

        return chain(place_iter, move_iter) if self.myMove >= 3 else move_iter

    @property
    def cells_around_hive(self) -> set[Cell]:
        """
        Set of all cells around the hive
        """
        return {
            neighbor
            for cell in self.hive
            for neighbor in self.empty_neighboring_cells(cell)
        }

        # Second option, I have to think about it more
        # visited: set[Cell] = set()
        #
        # for cell in self.hive:
        #     for neighbor in self.empty_neighboring_cells(cell):
        #         if neighbor not in visited:
        #             visited.add(neighbor)
        #             yield neighbor

    def move(self) -> MoveBrute:
        """
        Returns a best move for the current self.board. Has to first properly initialize
        the inner state.

        Format:
            [piece, oldP, oldQ, newP, newQ] - move from (oldP, oldQ) to (newP, newQ)
            [piece, None, None, newP, newQ] - place new at (newP, newQ)
            [] - no move is possible

        *Note: the API has to stay this way to be compatible with Brute*
        """

        self.hive = set(self.nonempty_cells)

        if not self.tournament:
            possible_moves = [*self.valid_moves]

            if not possible_moves:
                return []

            import random

            return random.choice(possible_moves).to_brute(self.myColorIsUpper)

        return []

    def moving_doesnt_break_hive(self, cell: Cell) -> bool:
        """
        Check if moving the given piece doesn't break the hive into parts
        """

        with LiftPiece(self, cell):
            queue = deque(self.neighbors(cell))
            visited: set[Cell] = set()

            consume(floodfill(visited, queue, self.neighbors, lambda _: None))

            return len(visited) == len(self.hive)

    def queens_moves(self, queen: Cell) -> Iterator[Move]:
        """
        Iterator over all valid moves for the queen in the current board.

        Queen can move one step in any direction.
        """

        assert self[queen].upper() == Piece.Queen

        move = functools.partial(Move, Piece.Queen, queen)

        return (move(target) for target in self.valid_steps(queen))

    def ants_moves(self, ant: Cell) -> Iterator[Move]:
        """
        Iterator over all valid moves for the ant in the current board.

        Ant can move any number of steps, but always has to stay right next
        to the hive.
        """

        with LiftPiece(self, ant) as piece:
            assert piece.upper() == Piece.Ant

            around_hive = set(self.cells_around_hive)

            def next_cells(cell: Cell) -> Iterator[Cell]:
                return (
                    neighbor
                    for neighbor in self.empty_neighboring_cells(cell)
                    if neighbor in around_hive and self.can_move_to(cell, neighbor)
                )

            move = functools.partial(Move, Piece.Ant, ant)
            visited: set[Cell] = {ant}
            queue = deque(next_cells(ant))

            yield from floodfill(visited, queue, next_cells, move)

    def beetles_moves(self, beetle: Cell) -> Iterator[Move]:
        """
        Iterator over all valid moves for the beetle in the current board.

        Beetle can make one step in any direction, while also being able to climb
        on top of other pieces.
        """

        assert self[beetle].upper() == Piece.Beetle

        move = functools.partial(Move, Piece.Beetle, beetle)

        return (
            move(target) for target in self.valid_steps(beetle, can_crawl_over=True)
        )

    def grasshoppers_moves(self, grasshopper: Cell) -> Iterator[Move]:
        """
        Iterator over all valid moves for the grasshopper in the current board.

        Grasshopper can jump in any direction in straght line, but always has to
        jump over at least one other piece.
        """

        assert self[grasshopper].upper() == Piece.Grasshopper

        move = functools.partial(Move, Piece.Grasshopper, grasshopper)

        # for each direction
        for dp, dq in DIRECTIONS:
            gs = grasshopper  # copy the position
            skipped = False

            # move in that direction until edge of board
            while True:
                p, q = gs
                gs = (p + dp, q + dq)

                if not self.in_board(gs):
                    break

                # if tile is not empty, skip the piece
                if not self.is_empty(gs):
                    skipped = True
                    continue

                # if tile is empty and at least one piece was skipped, yield move
                if skipped and self.has_neighbor(gs):
                    yield move(gs)

    def spiders_moves(self, spider: Cell) -> Iterator[Move]:
        """
        Iterator over all valid moves for the spider in the current board.

        Spider can move only exactly three steps, while staying right next
        to the hive.
        """

        assert self[spider].upper() == Piece.Spider

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

        return map(move, stack)

    def neighboring_cells(self, cell: Cell) -> Iterator[Cell]:
        """
        Iterator over all cells neighboring (p,q)
        """
        p, q = cell
        return (
            (p + dp, q + dq) for dp, dq in DIRECTIONS if self.in_board((p + dp, q + dq))
        )

    def empty_neighboring_cells(self, cell: Cell) -> Iterator[Cell]:
        """
        Iterator over all cells neighboring (p,q) that are empty
        """
        return (
            neighbor
            for neighbor in self.neighboring_cells(cell)
            if self.is_empty(neighbor)
        )

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        """
        Iterator over all cells neighboring (p,q) that aren't empty
        """
        return (
            neighbor
            for neighbor in self.neighboring_cells(cell)
            if not self.is_empty(neighbor)
        )

    def has_neighbor(self, cell: Cell, ignore: Cell | None = None) -> bool:
        """
        Check if the given cell has at least one neighbor
        """
        return any(neighbor for neighbor in self.neighbors(cell) if neighbor != ignore)

    def valid_steps(
        self,
        cell: Cell,
        can_crawl_over: bool = False,
    ) -> Iterator[Cell]:
        """
        Iterator over all cells neighboring (p,q), that can be accessed from (p,q) and
        moving to which won't leave the hive (that means they have at least
        one neighbor). By default, only empty cells are returned, but optionally
        non-empty cells can be included as well.
        """

        if can_crawl_over:
            return (
                neighbor
                for neighbor in self.neighboring_cells(cell)
                if self.has_neighbor(neighbor, ignore=cell)
            )

        return (
            neighbor
            for neighbor in self.empty_neighboring_cells(cell)
            if self.has_neighbor(neighbor, ignore=cell)
            and self.can_move_to(cell, neighbor)
        )

    def top_piece_in(self, cell: Cell) -> str:
        """
        Returns the top piece in given cell
        """
        return self[cell][-1]

    def is_my_cell(self, cell: Cell) -> bool:
        """
        Checks if (p,q) is a cell owned by the player
        """
        piece = self[cell][-1]
        is_upper = self.myColorIsUpper

        return piece.isupper() == is_upper or piece.islower() != is_upper

    def is_empty(self, cell: Cell) -> bool:
        """
        Checks if (p,q) is an empty cell
        """
        return self[cell] == ""

    def in_board(self, cell: Cell) -> bool:
        """
        Check if (p,q) is a valid coordinate within the board
        """
        p, q = cell
        return 0 <= q < self.size and -(q // 2) <= p < self.size - q // 2

    def rotate_left(self, direction: Direction) -> Direction:
        """
        Returns direction rotated one tile to left
        """
        p, q = direction
        return p + q, -p

    def rotate_right(self, direction: Direction) -> Direction:
        """
        Returns direction rotated one tile to right
        """
        p, q = direction
        return -q, p + q

    def neighbors_only_my_pieces(self, cell: Cell) -> bool:
        """
        Check if all neighbors of (p,q) are owned by the player
        """
        return all(self.is_my_cell(neighbor) for neighbor in self.neighbors(cell))

    def can_move_to(self, origin: Cell, target: Cell) -> bool:
        """
        Check if a piece can move from (p,q) to (np,nq), ie. there are no other pieces
        or board edges on the sides blocking it.
        """
        p, q = origin
        np, nq = target

        direction = (np - p, nq - q)

        lp, lq = self.rotate_left(direction)
        rp, rq = self.rotate_right(direction)

        left = (p + lp, q + lq)
        right = (p + rp, q + rq)

        return any(
            self.in_board(cell) and self.is_empty(cell) for cell in (left, right)
        )

    def remove_piece_from_board(self, cell: Cell) -> str:
        """
        Remove the top-most piece at the given cell
        """
        piece = self[cell][-1]
        self[cell] = self[cell][:-1]
        self.hive.remove(cell)
        return piece

    def add_piece_to_board(self, cell: Cell, piece: str) -> None:
        """
        Place the given piece at the given cell
        """
        self[cell] += piece
        self.hive.add(cell)

    def play_move(self, move: Move) -> None:
        """
        Play the given move
        """
        piece = move.piece_str(self.upper)
        start = move.start
        end = move.end

        # add the piece to its new position
        self.add_piece_to_board(end, piece)

        if start is not None:
            # remove the piece from its old position
            removed = self.remove_piece_from_board(start)
            assert removed == piece

    def reverse_move(self, move: Move) -> None:
        """
        Reverse the given move
        """
        piece = move.piece_str(self.upper)
        start = move.start
        end = move.end

        if start is not None:
            # add the piece back to its old position
            self.add_piece_to_board(start, piece)

        # remove the piece from its new position
        removed = self.remove_piece_from_board(end)
        assert removed == piece

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
                if self.in_board((p, q))
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
