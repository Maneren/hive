from typing import Any

from player import Player, convert_board, parse_board, play_move


def update_players(
    move: Any,
    active_player: Player,
    passive_player: Player,
) -> None:
    """Write move made by `active_player` player."""
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


def test_game() -> None:
    from time import perf_counter

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

    p1 = Player("player1", False, board_size, small_figures, big_figures)
    p2 = Player("player2", True, board_size, big_figures, small_figures)

    filename = "output/begin.png"
    p1.saveImage(filename)

    move_idx = 0
    while True:
        start = perf_counter()
        move1 = p1.move()
        elapsed = perf_counter() - start
        print("P1 returned", move1, "in", elapsed, "seconds")
        update_players(move1, p1, p2)  # update P1 and P2 according to the move
        p1.saveImage(f"output/move-{move_idx:03d}-player1.png")

        start = perf_counter()
        move2 = p2.move()
        elapsed = perf_counter() - start
        print("P2 returned", move2, "in", elapsed, "seconds")
        update_players(move2, p2, p1)  # update P2 and P1 according to the move
        p1.saveImage(f"output/move-{move_idx:03d}-player2.png")

        if not move1 and not move2:
            print(f"End of the test game after {move_idx} moves")
            break

        move_idx += 1
        p1.myMove = move_idx
        p2.myMove = move_idx

        if move_idx > 50:
            print("End of the test game")
            break


def test_position() -> None:
    board_size = 13
    small_figures = {
        "q": 0,
        "a": 0,
        "b": 0,
        "s": 0,
        "g": 0,
    }
    big_figures = {
        "Q": 0,
        "A": 0,
        "B": 0,
        "S": 0,
        "G": 0,
    }

    p = Player("player", True, board_size, big_figures, small_figures)

    from textwrap import dedent

    p.board = parse_board(
        dedent(
            """
            . . . . . . . . . . . . .
             . . . . . . . . . . . . .
            . . . . . s . . . . . . .
             . . . g s A . . . . . . .
            . . . . . . g . . . . . .
             . . . . . q . . . . . . .
            . . . . . b QB G S G . . .
             . . . . b . . S . a . . .
            . . . . . . . A . . . . .
             . . . . . . . B . . . . .
            . . . . . . . . a . . . .
             . . . . . . . . . . . . .
            . . . . . . . . . . . . .
            """,
        ).strip(),
    )

    p._board = convert_board(p.board)
    p.hive = set(p.nonempty_cells)
    p.myMove = 6

    filename = "output/begin.png"
    p.saveImage(filename)

    for i, move in enumerate(list(p.valid_moves)):
        print(move)
        with play_move(p, move):
            filename = f"output/move-{i:03d}.png"
            p.saveImage(filename)

    print("End of found moves")

    move_brute = p.move()
    print("P returned", move_brute)

    # print("//////////")

    # from michal import Player as MPlayer
    #
    # mp = MPlayer("michal", False, board_size, small_figures, big_figures)
    # mp.board = p.board
    #
    # for move in mp.allMyMoves(5):
    #     mm = move.moves
    #     for m in mm:
    #         print(Move(move.name, (move.p, move.q) if move.p is not None else None, m))


if __name__ == "__main__":
    test_position()
