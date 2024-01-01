from __future__ import annotations

import functools
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from itertools import chain
from typing import Any, Callable, Iterator, TypeVar

from base import Board

TEST = False

# Player template for HIVE --- ALP semestral work
# Vojta Vonasek, 2023


# PUT ALL YOUR IMPLEMENTATION INTO THIS FILE

BoardData = dict[int, dict[int, list[str]]]
BoardDataBrute = dict[int, dict[int, str]]
Cell = tuple[int, int]
Direction = tuple[int, int]
MoveBrute = list[str, int, int, int, int] | list[str, None, None, int, int]

DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

T = TypeVar("T")


def count(iterator: Iterator[T]) -> int:
    """
    Count the number of elements in an iterator
    """
    return sum(1 for _ in iterator)


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


def parse_board(string: str) -> BoardDataBrute:
    """
    Parse board from string
    """
    board: BoardDataBrute = {}
    lines = string.splitlines()
    for q, line in enumerate(lines):
        for p, char in enumerate(line.split(), start=-q // 2):
            if p not in board:
                board[p] = {}

            board[p][q] = char if char != "." else ""

    return board


def convert_board(board: BoardDataBrute) -> BoardData:
    return {p: {q: list(board[p][q]) for q in board[p]} for p in board}


class Criteria(IntEnum):
    BASE = 0
    ANT_BLOCKING = 1
    BEETLE_BLOCKING = 2
    BEETLE_BLOCKING_QUEEN = 3
    QUEEN_NEIGHBOR = 4
    QUEEN_SURROUNDED = 5
    QUEEN_BLOCKED = 6
    SPIDER_BLOCKING = 7


EVAL_TABLE_MY = [1, 600, 200, 1000, -400, -1000000, -400, 200]
EVAL_TABLE_RIVAL = [-1, -500, -200, -1200, 400, 1000000, 400, -100]


def evaluate_cell(
    player: Player,
    cell: Cell,
) -> tuple[int, State]:
    my = player.is_my_cell(cell)
    piece = Piece.from_str(player.top_piece_in(cell))

    table = EVAL_TABLE_MY if my else EVAL_TABLE_RIVAL

    score = table[Criteria.BASE]

    if piece == Piece.Ant:
        if count(player.neighbors(cell)) == 1:
            score = table[Criteria.ANT_BLOCKING]
    elif piece == Piece.Beetle:
        if len(player[cell]) == 2:
            score = table[Criteria.BEETLE_BLOCKING]
            if player[cell][0].upper() == Piece.Queen:
                score += table[Criteria.BEETLE_BLOCKING_QUEEN]
    elif piece == Piece.Queen:
        c = count(player.neighbors(cell))

        if c == 6:
            return table[Criteria.QUEEN_SURROUNDED], State.LOSS if my else State.WIN

        score = table[Criteria.QUEEN_NEIGHBOR] * (c - 1)

        if player.moving_breaks_hive(cell):
            score += table[Criteria.QUEEN_BLOCKED]

    return score, State.RUNNING


evaluated = 0


def evaluate_position(player: Player) -> tuple[int, State]:
    """
    Evaluate the position from the POV of the given player
    """

    global evaluated

    evaluated += 1

    score = 0

    for cell in player.hive:
        piece_score, game_state = evaluate_cell(player, cell)
        if game_state:
            return piece_score, game_state
        score += piece_score

    return score, State.RUNNING


class Piece(str, Enum):
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
    """
    Holds a move - piece, start cell and end cell. Start is None for placing a new piece
    """

    _piece: Piece
    start: Cell | None
    end: Cell

    def to_brute(self, upper: bool) -> MoveBrute:
        """
        Convert the move to brute representation
        """
        piece = self.piece_str(upper)

        if self.start is None:
            return [piece, None, None, *self.end]

        return [piece, *self.start, *self.end]

    def piece_str(self, upper: bool) -> str:
        """
        Return the string representation of the used piece
        """
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
    """
    Node in the minimax tree.
    """

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
        self.move = move
        self.player = player
        self.score = 0
        self.children = []
        self.depth = 0
        self.state = State.RUNNING

    def next_depth(self, player: Player) -> None:
        """
        Compute the next depth using the minimax algorithm
        """
        if self.state.is_end():
            return

        self.depth += 1

        with PlayMove(player, self.move):
            if self.depth == 1:
                self.score, self.end = evaluate_position(self.player)
                return

            if self.depth == 2:
                self.children = [Node(move, player) for move in player.valid_moves]
            else:
                for child in self.children:
                    child.next_depth(player)

            self.score = self.evaluate_children()

    def evaluate_children(self) -> int:
        """
        Evaluate the children of the current node
        """

        if not self.children:
            self.state = State.DRAW
            return 0

        children = self.children

        children.sort(reverse=True)

        depth = self.depth
        if depth <= 4:
            limit = 10
        elif depth <= 6:
            limit = 5
        else:
            limit = 2

        self.children = children[:limit]

        return -self.children[0].score

    def __gt__(self, other: Node) -> bool:
        return self.score > other.score

    def __str__(self) -> str:
        return f"{self.move}: {self.score}"


class LiftPiece:
    """
    Lifts a piece from the board for the duration of the context
    """

    player: Player
    cell: Cell

    piece: str

    def __init__(self, player: Player, cell: Cell) -> None:
        self.player = player
        self.cell = cell

    def __enter__(self) -> str:
        self.piece = self.player.remove_piece_from_board(self.cell)
        return self.piece

    def __exit__(self, *args: Any) -> None:
        self.player.add_piece_to_board(self.cell, self.piece)


class PlayMove:
    """
    Plays a move for the duration of the context
    """

    player: Player
    move: Move

    def __init__(self, player: Player, move: Move) -> None:
        self.player = player
        self.move = move

    def __enter__(self) -> None:
        self.player.play_move(self.move)

    def __exit__(self, *args: Any) -> None:
        self.player.reverse_move(self.move)


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
        Iterator over all my pieces on the board. Uses self.hive directly
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
        Iterator over all my movable pieces. Uses self.hive transitively
        """
        return (
            (piece, cell)
            for piece, cell in self.my_pieces_on_board
            if not self.moving_breaks_hive(cell)
        )

    @property
    def valid_placements(self) -> Iterator[Cell]:
        """
        Iterator over all valid placements. Uses self.hive transitively
        and expects at least one piece to be already placed
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

        is_my_queen_on_board = self.my_piece_remaining(Piece.Queen) == 0

        if not is_my_queen_on_board and self.myMove == 4:
            return map(
                functools.partial(Move, Piece.Queen, None),
                self.valid_placements,
            )

        return chain(place_iter, move_iter) if is_my_queen_on_board else place_iter

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

        import random
        import time

        start = time.perf_counter()

        self.hive = set(self.nonempty_cells)
        self._board = convert_board(self.board)

        if self.myMove == 0:
            if not self.hive:
                return Move(Piece.Spider, None, (1, 6)).to_brute(self.upper)

            placement = random.choice(list(self.cells_around_hive))
            return Move(Piece.Spider, None, placement).to_brute(self.upper)

        if TEST:
            possible_moves = list(self.valid_moves)

            if not possible_moves:
                return []

            return random.choice(possible_moves).to_brute(self.upper)

        nodes = [Node(move, self) for move in self.valid_moves]

        if not nodes:
            return []

        depth = 0

        while time.perf_counter() - start < 0.9:
            depth += 1

            for node in nodes:
                node.next_depth(self)

            if depth <= 2:
                limit = len(nodes)
            elif depth <= 5:
                limit = 5
            else:
                limit = 2

            if limit < len(nodes):
                nodes.sort(reverse=True)
                nodes = nodes[:limit]

        best = max(nodes)

        end = time.perf_counter()

        elapsed = end - start

        print(f"Searched to depth {depth} ({evaluated} positions) in {elapsed} seconds")

        return best.move.to_brute(self.upper)

    def moving_breaks_hive(self, cell: Cell) -> bool:
        """
        Check if moving the given piece breaks the hive into parts
        """

        with LiftPiece(self, cell):
            start = next(self.neighbors(cell), None)
            if not start:
                return False

            queue = deque([start])
            visited: set[Cell] = set()

            # consume the iterator to visit all the cells in the hive
            _ = list(floodfill(visited, queue, self.neighbors, lambda _: None))

            return len(visited) != len(self.hive)

    def queens_moves(self, queen: Cell) -> Iterator[Move]:
        """
        Iterator over all valid moves for the queen in the current board.

        Queen can move one step in any direction.
        """

        with LiftPiece(self, queen) as piece:
            assert piece.upper() == Piece.Queen, f"{piece} at {queen} is not a Queen"
            move = functools.partial(Move, Piece.Queen, queen)
            yield from map(move, self.valid_steps(queen))

    def ants_moves(self, ant: Cell) -> Iterator[Move]:
        """
        Iterator over all valid moves for the ant in the current board.

        Ant can move any number of steps, but always has to stay right next
        to the hive.
        """

        with LiftPiece(self, ant) as piece:
            assert piece.upper() == Piece.Ant, f"{piece} at {ant} is not an Ant"

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

        with LiftPiece(self, beetle) as piece:
            assert piece.upper() == Piece.Beetle, f"{piece} at {beetle} is not a Beetle"

            move = functools.partial(Move, Piece.Beetle, beetle)

            yield from map(move, self.valid_steps(beetle, can_crawl_over=True))

    def grasshoppers_moves(self, grasshopper: Cell) -> Iterator[Move]:
        """
        Iterator over all valid moves for the grasshopper in the current board.

        Grasshopper can jump in any direction in straght line, but always has to
        jump over at least one other piece.
        """

        with LiftPiece(self, grasshopper) as piece:
            assert (
                piece.upper() == Piece.Grasshopper
            ), f"{piece} at {grasshopper} is not a Grasshopper"

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

                    # if tile is empty and something was skipped, yield move
                    # else try different direction
                    if self.is_empty(gs):
                        if skipped:
                            yield move(gs)

                        break

                    # if tile is not empty, skip the piece
                    skipped = True

    def spiders_moves(self, spider: Cell) -> Iterator[Move]:
        """
        Iterator over all valid moves for the spider in the current board.

        Spider can move only exactly three steps, while staying right next
        to the hive.
        """

        with LiftPiece(self, spider) as piece:
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

    def valid_steps(
        self,
        cell: Cell,
        /,
        can_crawl_over: bool = False,
    ) -> Iterator[Cell]:
        """
        Iterator over all cells neighboring (p,q), that can be accessed from (p,q) and
        moving to which won't leave the hive (that means they have at least
        one neighbor). By default, only empty cells are returned, but optionally
        non-empty cells can be included as well.
        """
        return (
            neighbor
            for neighbor in self.neighboring_cells(cell)
            if self.can_move_to(cell, neighbor, can_crawl_over=can_crawl_over)
        )

    def top_piece_in(self, cell: Cell) -> str:
        """
        Returns the top piece in given cell
        """
        return self[cell][-1]

    def my_piece_remaining(self, piece: Piece) -> int:
        """
        Returns the number of my pieces of given type
        """
        piece_str = piece.upper() if self.myColorIsUpper else piece.lower()
        return self.myPieces[piece_str]

    def is_my_cell(self, cell: Cell) -> bool:
        """
        Checks if (p,q) is a cell owned by the player
        """
        piece = self[cell][-1]

        return piece.isupper() == self.upper or piece.islower() != self.upper

    def is_empty(self, cell: Cell) -> bool:
        """
        Checks if (p,q) is an empty cell
        """
        return not self[cell]

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

    def can_move_to(
        self,
        origin: Cell,
        target: Cell,
        /,
        can_crawl_over: bool = False,
    ) -> bool:
        """
        Check if a piece can move from (p,q) to (np,nq), ie. there is exactly one
        neighbor in the direction of move
        """
        p, q = origin
        np, nq = target

        direction = (np - p, nq - q)

        lp, lq = self.rotate_left(direction)
        rp, rq = self.rotate_right(direction)

        left = (p + lp, q + lq)
        right = (p + rp, q + rq)

        def is_full(cell: Cell) -> bool:
            return not (self.is_empty(cell) or self.in_board(cell))

        left_full, right_full = map(is_full, [left, right])

        return left_full or right_full if can_crawl_over else left_full != right_full

    def remove_piece_from_board(self, cell: Cell) -> str:
        """
        Remove the top-most piece at the given cell and return it
        """
        piece = self[cell].pop()

        if self.is_empty(cell):
            self.hive.remove(cell)

        return piece

    def add_piece_to_board(self, cell: Cell, piece: str) -> None:
        """
        Place the given piece at the given cell
        """
        self[cell].append(piece)
        self.hive.add(cell)

    def play_move(self, move: Move) -> None:
        """
        Play the given move
        """
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
    def __getitem__(self, cell: Cell) -> list[str]:
        p, q = cell
        return self._board[p][q]

    def __setitem__(self, cell: Cell, value: str | list[str]) -> None:
        p, q = cell
        self._board[p][q] = value if isinstance(value, list) else list(value)

    def __str__(self) -> str:
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
