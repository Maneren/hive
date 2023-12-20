from __future__ import annotations
import hive.base as Base
from typing import Generator, TypeAlias
from enum import Enum


# Player template for HIVE --- ALP semestral work
# Vojta Vonasek, 2023


# PUT ALL YOUR IMPLEMENTATION INTO THIS FILE

BoardData: TypeAlias = dict[int, dict[int, str]]
Tile: TypeAlias = tuple[int, int]
TilesGenerator: TypeAlias = Generator[Tile, None, None]
Move: TypeAlias = list[str, int, int, int,
                       int] | list[str, None, None, int, int]


def play_move(board: BoardData, move: Move) -> BoardData:
    piece, fromP, fromQ, toP, toQ = move

    # add the piece to its new position
    board[toP][toQ] = board[toP][toQ] + piece

    if fromP is not None:
        # remove the piece from its old position
        assert board[fromP][fromQ][-1] == piece
        board[fromP][fromQ] = board[fromP][fromQ][:-1]


def reverse_move(board: BoardData, move: Move) -> None:
    piece, fromP, fromQ, toP, toQ = move

    if fromP is not None:
        # add the piece back to its old position
        board[fromP][fromQ] = board[fromP][fromQ] + board[toP][toQ]

    # remove the piece from its new position
    assert board[toP][toQ][-1] == piece
    board[toP][toQ] = board[toP][toQ][:-1]


class Node:
    class State(Enum):
        RUNNING = 0
        WIN = 1
        LOSS = 2
        DRAW = 3

        def is_end(self) -> bool:
            return self != Node.State.RUNNING

    move: Move
    player_is_upper: bool
    score: int
    children: list[Node]
    depth: int
    state: Node.State

    def __init__(self, move: Move, player_is_upper: bool, initial_score: int) -> None:
        self.move = move
        self.player_is_upper = player_is_upper
        self.score = initial_score
        self.children = []
        self.depth = 0
        self.state = Node.State.RUNNING


class Player(Base.Board):
    def __init__(
        self,
        playerName: str,
        myIsUpper: bool,
        size: int,
        myPieces: dict[str, int],
        rivalPieces: dict[str, int],
    ):
        """
        Do not change this method
        """
        Base.Board.__init__(self, myIsUpper, size, myPieces, rivalPieces)
        self.playerName = playerName
        self.algorithmName = "maneren"

    def empty_cells_iter(self) -> TilesGenerator:
        yield from (
            (p, q)
            for p in self.board
            for q in self.board[p]
            if self.isEmpty(p, q, self.board)
        )

    def nonempty_cells_iter(self) -> TilesGenerator:
        yield from (
            (p, q)
            for p in self.board
            for q in self.board[p]
            if not self.isEmpty(p, q, self.board)
        )

    def my_pieces_iter(self) -> Generator[tuple[str, Tile], None, None]:
        for x, y in self.nonempty_cells_iter():
            piece = self.board[x][y]
            if piece.isupper() == self.myColorIsUpper:
                yield piece, (x, y)

    def move(self) -> Move | []:
        """
        return [animal, oldP, oldQ, newP, newQ],
        or [animal, None, None, newP, newQ]
        or [] if no move is possible

        this has to stay the same for compatibility with BRUTE
        """

        score, best_move = minimax(
            self, self.myPieces, self.rivalPieces, [], 0)

    def neighbor_tiles_iter(self, p: int, q: int) -> TilesGenerator:
        """
        Iterator over all tiles neighboring the tile (p,q)
        in the hexagonal board
        """
        # we have 3 possible axis
        for i in range(3):
            # two neighbors on each
            for j in [-1, 1]:
                # for axis 0 and 2 we modify p, and for 1 q
                if i % 2 == 0:
                    x = p + j
                    y = q
                else:
                    x = p
                    y = q + j

                if self.inBoard(x, y):
                    yield x, y

    def empty_neighbor_tiles_iter(self, p: int, q: int) -> TilesGenerator:
        return (
            tile
            for tile in self.neighbor_tiles_iter(p, q)
            if self.isEmpty(*tile, self.board)
        )

    def is_my_tile(self, p: int, q: int) -> bool:
        return self.board[p][q].isupper() == self.myColorIsUpper


def is_valid_move(board: BoardData, piece: str, x: int, y: int) -> bool:
    return board[x][y][-1] == piece


def is_valid_initial_placement(player: Player, piece: str, p: int, q: int) -> bool:
    neighboring_my = False
    neighboring_opponent = False

    for p, q in player.empty_neighbor_tiles_iter(p, q):
        if player.is_my_tile(p, q):
            neighboring_my = True
        else:
            # initial placement can't be neighboring opponent's pieces
            return False

    return neighboring_my and not neighboring_opponent


def minimax(
    player: Player,
    my_pieces: dict[str, int],
    opponents_pieces: dict[str, int],
    move: int,
    depth: int,
    if depth > 5:
    alpha: int = -(10**9),
    beta: int = 10**9,
) -> tuple[int, Move]:
        return

    for piece, (piece_x, piece_y) in player.my_pieces_iter():
        for x, y in player.empty_cells_iter():
            if not is_valid_move(player.board, piece, x, y):
                continue

            print(piece, piece_x, piece_y, x, y)


def updatePlayers(move, activePlayer, passivePlayer):
    """write move made by activePlayer player
    this method assumes that all moves are correct, no checking is made
    """
    if not move:
        return

    animal, p, q, newp, newq = move
    if p is None and q is None:
        # placing new animal
        activePlayer.myPieces[animal] -= 1
        passivePlayer.rivalPieces = activePlayer.myPieces.copy()
    else:
        # just moving animal
        # delete its old position
        activePlayer.board[p][q] = activePlayer.board[p][q][:-1]
        passivePlayer.board[p][q] = passivePlayer.board[p][q][:-1]

    activePlayer.board[newp][newq] += animal
    passivePlayer.board[newp][newq] += animal
