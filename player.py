from __future__ import annotations
import base as Base
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


def play_move(board: BoardData, move: Move) -> None:
    piece, from_p, from_q, to_p, to_q = move

    # add the piece to its new position
    board[to_p][to_q] = board[to_p][to_q] + piece

    if from_p is not None:
        # remove the piece from its old position
        assert board[from_p][from_q][-1] == piece
        board[from_p][from_q] = board[from_p][from_q][:-1]


def reverse_move(board: BoardData, move: Move) -> None:
    piece, from_p, from_q, to_p, to_q = move

    if from_p is not None:
        # add the piece back to its old position
        board[from_p][from_q] = board[from_p][from_q] + board[to_p][to_q]

    # remove the piece from its new position
    assert board[to_p][to_q][-1] == piece
    board[to_p][to_q] = board[to_p][to_q][:-1]


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

    def __init__(
        self, move: Move, player_is_upper: bool, initial_score: int = 0
    ) -> None:
        self.move = move
        self.player_is_upper = player_is_upper
        self.score = initial_score
        self.children = []
        self.depth = 0
        self.state = Node.State.RUNNING

    def next_depth(self, player: Player) -> None:
        assert not self.state.is_end()

        self.depth += 1

        if self.depth == 1:
            self.initialize_children(player)
            return

        play_move(player.board, self.move)

        for child in self.children:
            child.next_depth(player)

        reverse_move(player.board, self.move)

    def initialize_children(self, player: Player) -> None:
        play_move(player.board, self.move)

        for move in player.valid_moves():
            self.children.append(Node(move, not self.player_is_upper))

        reverse_move(player.board, self.move)


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

    def cells(self) -> TilesGenerator:
        """
        Iterator over all cells
        """
        yield from ((p, q) for p in self.board for q in self.board[p])

    def empty_cells(self) -> TilesGenerator:
        """
        Iterator over all empty cells
        """
        board = self.board
        yield from (tile for tile in self.cells() if self.isEmpty(*tile, board))

    def nonempty_cells(self) -> TilesGenerator:
        """
        Iterator over all nonempty cells
        """
        yield from (
            tile for tile in self.cells() if not self.isEmpty(*tile, self.board)
        )

    def my_pieces(self) -> Generator[tuple[str, Tile], None, None]:
        """
        Iterator over all my pieces on the board
        """
        board = self.board

        yield from (
            (board[p][q], (p, q))
            for p, q in self.my_cells_iter()
            if self.isMyColor(p, q, board)
        )

    def move(self) -> Move | []:
        """
        return [animal, oldP, oldQ, newP, newQ],
        or [animal, None, None, newP, newQ]
        or [] if no move is possible

        this has to stay the same for compatibility with BRUTE
        """

        return []

    def neighbor_tiles(self, p: int, q: int) -> TilesGenerator:
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

    def empty_neighbor_tiles(self, p: int, q: int) -> TilesGenerator:
        return (
            tile
            for tile in self.neighbor_tiles(p, q)
            if self.isEmpty(*tile, self.board)
        )


def is_valid_move(board: BoardData, piece: str, x: int, y: int) -> bool:
    return board[x][y][-1] == piece


def is_valid_initial_placement(player: Player, piece: str, p: int, q: int) -> bool:
    neighboring_my = False
    neighboring_opponent = False

    for p, q in player.empty_neighbor_tiles(p, q):
        if player.is_my_tile(p, q):
            neighboring_my = True
        else:
            # initial placement can't be neighboring opponent's pieces
            return False

    return neighboring_my and not neighboring_opponent


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


def main():
    boardSize = 13
    smallFigures = {
        "q": 1,
        "a": 2,
        "b": 2,
        "s": 2,
        "g": 2,
    }  # key is animal, value is how many is available for placing
    bigFigures = {
        figure.upper(): smallFigures[figure] for figure in smallFigures
    }  # same, but with upper case

    P1 = Player("player1", True, 13, smallFigures, bigFigures)
    P2 = Player("player2", True, 13, bigFigures, smallFigures)

    filename = "output/begin.png"
    P1.saveImage(filename)

    moveIdx = 0
    while True:
        move = P1.move()
        print("P1 returned", move)
        updatePlayers(move, P1, P2)  # update P1 and P2 according to the move
        filename = "output/move-{:03d}-player1.png".format(moveIdx)
        P1.saveImage(filename)

        move = P2.move()
        print("P2 returned", move)
        updatePlayers(move, P2, P1)  # update P2 and P1 according to the move
        filename = "output/move-{:03d}-player2.png".format(moveIdx)
        P1.saveImage(filename)

        moveIdx += 1
        P1.myMove = moveIdx
        P2.myMove = moveIdx

        if moveIdx > 50:
            print("End of the test game")
            break


if __name__ == "__main__":
    main()
