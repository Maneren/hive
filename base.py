from __future__ import annotations

import math
from typing import TypeAlias

from PIL import Image, ImageDraw

# DO NOT MODIFY THIS FILE
# THIS FILE IS NOT UPLOADED TO BRUTE (all changes in it will be ignored by Brute)

BoardData: TypeAlias = dict[int, dict[int, str]]
Tile: TypeAlias = tuple[int, int]
PiecesDict: TypeAlias = dict[str, int]


class Board:
    size: int
    myMove: int
    board: BoardData
    myColorIsUpper: bool
    algorithmName: str
    playerName: str

    myPieces: PiecesDict
    rivalPieces: PiecesDict

    def __init__(
        self,
        myIsUpper: bool,
        size: int,
        myPieces: PiecesDict,
        rivalPieces: PiecesDict,
    ) -> None:
        self.size = size  # size of the board
        self.myMove = 0  # index of actual move
        self.myColorIsUpper = myIsUpper
        self.algorithmName = "some algorithm"
        self.playerName = "some name"
        self.tournament = (
            0  # filled by Brute, if True, player is run in tournament mode
        )

        # dict of key=animal, value = number of available pieces of given type
        self.myPieces = myPieces.copy()
        self._myPiecesOriginal = myPieces.copy()

        self.rivalPieces = rivalPieces.copy()
        self._rivalPiecesOriginal = rivalPieces.copy()

        # the rest of the code is just for drawing to png

        image_names = ["ant", "beetle", "bee", "spider", "grasshopper"]

        self._images = {
            name: Image.open(f"images/{name}.png").resize((70, 70))
            for name in image_names
        }

        self._images_small = {
            name: Image.open(f"images/{name}.png").resize((20, 20))
            for name in image_names
        }

        # create empty board as a dictionary
        self.board = {
            p: {q: "" for q in range(-size, size) if self.inBoard(p, q)}
            for p in range(-size, size)
        }

        # this is for visualization and to synchronize colors between png/js
        self._colors = {
            -1: "#fdca40",
            0: "#ffffff",
            1: "#947bd3",
            2: "#ff0000",
            3: "#00ff00",
            4: "#0000ff",
            5: "#566246",
            6: "#a7c4c2",
            7: "#ADACB5",
            8: "#8C705F",
            9: "#FA7921",
            10: "#566E3D",
        }

    def inBoard(self, p: int, q: int) -> bool:
        """return True if (p,q) is valid coordinate"""
        return 0 <= q < self.size and -q // 2 <= p < self.size - q // 2

    def rotateRight(self, p: int, q: int) -> Tile:
        pp = -q
        qq = p + q
        return pp, qq

    def rotateLeft(self, p: int, q: int) -> Tile:
        pp = p + q
        qq = -p
        return pp, qq

    def letter2image(self, last_letter: str) -> tuple[Image.Image, Image.Image]:
        last_letter = last_letter.lower()

        image_names_map = {
            "b": "beetle",
            "s": "spider",
            "g": "grasshopper",
            "q": "bee",
            "a": "ant",
        }

        if last_letter not in image_names_map:
            return None, None

        image_name = image_names_map[last_letter]

        return self._images[image_name], self._images_small[image_name]

    def saveImage(self, filename, HL={}, LINES=[], HLA={}):
        """draw actual board to png. Empty cells are white, -1 = red,
        1 = green, other values according to this list:
        -1 red, 0 = white, 1 = green

        HL is dict of coordinates and colors, e.g.
        HL[(3,4)] = #RRGGBB #will use color #RRGGBB to highlight cell (3,4)
        LINES is list of extra lines to be drawn in format
        LINES = [ line1, line2 ,.... ], where each line is [#RRGGBB, p1, q1, p2, q2]
        - will draw line from cell (p1,q1) to cell (p2,q2)
        """

        def pq2hexa(p, q):
            cx = cellRadius * (math.sqrt(3) * p + math.sqrt(3) / 2 * q) + cellRadius
            cy = cellRadius * (0 * p + 3 / 2 * q) + cellRadius

            pts = []
            for a in [30, 90, 150, 210, 270, 330]:
                nx = cx + cellRadius * math.cos(a * math.pi / 180)
                ny = cy + cellRadius * math.sin(a * math.pi / 180)
                pts.append(nx)
                pts.append(ny)
            return cx, cy, pts

        def drawPieces(piecesToDraw, piecesToDrawOriginal, draw, p, q, HLA={}):
            # HLA is dict, key is animal for highlight, value is color,
            # e.g. HLA["a"] = "#RRGGBB" will highlight my own piece 'a'
            for animal in piecesToDraw:
                for v in range(piecesToDrawOriginal[animal]):
                    # draw this animal
                    cx, cy, pts = pq2hexa(p, q)
                    color = "#ff00ff"

                    lastLetter = animal
                    color = (
                        self._colors[-1] if lastLetter.islower() else self._colors[1]
                    )
                    if v < piecesToDraw[animal] and animal in HLA:
                        color = HLA[animal]

                    draw.polygon(pts, fill=color)
                    pts.append(pts[0])
                    pts.append(pts[1])
                    draw.line(pts, fill="black", width=1)

                    lastLetter = animal.lower()
                    icx = int(cx) - cellRadius // 1
                    icy = int(cy) - cellRadius // 1
                    if v < piecesToDraw[animal]:
                        impaste, impaste2 = self.letter2image(lastLetter)
                        if impaste:
                            img.paste(impaste, (int(icx), int(icy)), impaste)
                    p += 1

        cellRadius = 35
        cellWidth = int(cellRadius * (3**0.5))
        cellHeight = 2 * cellRadius

        width = cellWidth * self.size + cellRadius * 3
        height = cellHeight * self.size

        img = Image.new("RGB", (width, height), "white")

        draw = ImageDraw.Draw(img)

        allQ = []
        allP = []

        for p in self.board:
            allP.append(p)
            for q in self.board[p]:
                allQ.append(q)
                cx, cy, pts = pq2hexa(p, q)

                color = "#ff00ff"  # pink is for values out of range -1,..10
                if self.isEmpty(p, q, self.board):
                    color = self._colors[0]
                else:
                    lastLetter = self.board[p][q][-1]
                    color = (
                        self._colors[-1] if lastLetter.islower() else self._colors[1]
                    )

                if (p, q) in HL:
                    color = HL[(p, q)]
                draw.polygon(pts, fill=color)

                if (
                    not self.isEmpty(p, q, self.board)
                    and self.board[p][q][-1].lower() in "bB"
                    and len(self.board[p][q]) > 1
                ):
                    # draw half of the polygon in red color to highlight that beetle is on the top
                    polygon2 = pts[6:] + pts[:2]
                    draw.polygon(polygon2, fill=self._colors[2])

                pts.append(pts[0])
                pts.append(pts[1])
                draw.line(pts, fill="black", width=1)
                draw.text([cx - 3, cy - 3], f"{p} {q}", fill="black", anchor="mm")
                if not self.isEmpty(p, q, self.board):
                    draw.text(
                        [cx, cy],
                        f"{self.board[p][q]}",
                        fill="black",
                        anchor="mm",
                    )
                    lastLetter = self.board[p][q][-1].lower()

                    icx = int(cx) - cellRadius // 1
                    icy = int(cy) - cellRadius // 1
                    impaste, impaste2 = self.letter2image(lastLetter)

                    if impaste:
                        img.paste(impaste, (int(icx), int(icy)), impaste)

        maxq = max(allQ)
        minp = min(allP)
        maxq += 2
        minp += 1

        drawPieces(self.myPieces, self._myPiecesOriginal, draw, minp, maxq, HLA)
        maxq += 1
        drawPieces(self.rivalPieces, self._rivalPiecesOriginal, draw, minp, maxq, HLA)

        for line in LINES:
            color, p1, q1, p2, q2 = line
            cx1, cy1, _ = pq2hexa(p1, q1)
            cx2, cy2, _ = pq2hexa(p2, q2)
            draw.line([cx1, cy1, cx2, cy2], fill=color, width=2)

        img.save(filename)

    def print(self, board: BoardData) -> None:
        for p in board:
            for q in board[p]:
                value = board[p][q]
                print(value or "..", end="  ")
            print()

    def isMyColor(self, p: int, q: int, board: BoardData) -> bool:
        """assuming board[p][q] is not empty"""
        return (
            board[p][q].isupper() == self.myColorIsUpper
            or board[p][q].islower() != self.myColorIsUpper
        )

    def isEmpty(self, p: int, q: int, board: dict[int, dict[int, str]]) -> bool:
        return board[p][q] == ""

    def a2c(self, p: int, q: int) -> tuple[int, int, int]:
        x = p
        z = q
        y = -x - z
        return x, y, z

    def c2a(self, x: int, _y: int, z: int) -> tuple[int, int]:
        p = x
        q = z
        return p, q

    def distance(self, p1: int, q1: int, p2: int, q2: int) -> int:
        """return distance between two cells (p1,q1) and (p2,q2)"""
        x1, y1, z1 = self.a2c(p1, q1)
        x2, y2, z2 = self.a2c(p2, q2)
        return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) // 2
