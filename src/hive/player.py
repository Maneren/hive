import hive.base as Base
import random
from typing import Generator, TypeAlias


# Player template for HIVE --- ALP semestral work
# Vojta Vonasek, 2023


# PUT ALL YOUR IMPLEMENTATION INTO THIS FILE

BoardData: TypeAlias = dict[int, dict[int, str]]
Tile: TypeAlias = tuple[int, int]
TilesGenerator: TypeAlias = Generator[Tile, None, None]


class Player(Base.Board):
    def __init__(
        self,
        playerName: str,
        myIsUpper: bool,
        size: int,
        myPieces: dict[str, int],
        rivalPieces: dict[str, int],
    ):  # do not change this line
        Base.Board.__init__(
            self, myIsUpper, size, myPieces, rivalPieces
        )  # do not change this line
        self.playerName = playerName
        self.algorithmName = "maneren"

    def empty_cells_iter(self) -> TilesGenerator:
        for p in self.board:
            for q in self.board[p]:
                if self.isEmpty(p, q, self.board):
                    yield p, q

    def nonempty_cells_iter(self) -> TilesGenerator:
        for p in self.board:
            for q in self.board[p]:
                if not self.isEmpty(p, q, self.board):
                    yield p, q

    def my_pieces_iter(self) -> Generator[tuple[str, Tile], None, None]:
        for x, y in self.nonempty_cells_iter():
            piece = self.board[x][y]
            if piece.isupper() == self.myColorIsUpper:
                yield piece, (x, y)

    def move(self):
        """
        return [animal, oldP, oldQ, newP, newQ],
        or [animal, None, None, newP, newQ]
        or []
        """

        # the following code just randomly places (ignoring all the rules) some
        # random figure at the board
        emptyCells = [*self.empty_cells_iter()]

        if len(emptyCells) == 0:
            return []

        randomCell = emptyCells[random.randint(0, len(emptyCells) - 1)]
        randomP, randomQ = randomCell

        for animal in self.myPieces:
            if (
                self.myPieces[animal] > 0
            ):  # is this animal still available? if so, let's place it
                return [animal, None, None, randomP, randomQ]

        # all animals are places, let's move some randomly
        # (again, while ignoring all rules)
        allFigures = self.getAllNonemptyCells()
        randomCell = allFigures[random.randint(0, len(allFigures) - 1)]
        randomFigureP, randomFigureQ = randomCell
        # determine which animal is at randomFigureP, randomFigureQ
        animal = self.board[randomFigureP][randomFigureQ][
            -1
        ]  # [-1] means the last letter
        return [animal, randomFigureP, randomFigureQ, randomP, randomQ]

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


# def is_valid_placement(player: Player, piece: str, p: int, q: int) -> bool:
#     neighboring_my = False
#     neighboring_opponent = False
#
#     for p, q in player.empty_neighbor_tiles_iter(p, q):
#         if player.board[p][q].isupper() == player.myColorIsUpper:
#             neighboring_my = True
#         else:
#             neighboring_opponent = True
#
#     # return all(
#     #     lambda (p, q): board[p][q].isupper() == board.myColorIsUpper, board.empty_neighbor_tiles_iter(p, q))
#
#
def minimax_movement(
    player: Player,
    my_pieces: dict[str, int],
    opponents_pieces: dict[str, int],
    move: int,
    depth: int,
    alpha: int,
    beta: int,
):
    if depth > 5:
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
