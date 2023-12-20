from hive.player import Player, updatePlayers


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
