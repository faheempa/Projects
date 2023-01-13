from tkinter import *
from button_class import *
import time

def start_bot(i, j):
    bot = "O"
    user = "X"
    target = i * 3 + j
    clicked_btn = buttons[target]
    clicked_btn.set_text(user)
    board = [btn.text for btn in buttons]
    if win(user, board):
        print_win_or_tie("YOU---WIN")
    elif tie(board):
        print_win_or_tie("---TIE---")
    else:
        best_pos = minmax_helper(True, user, bot, board)
        board[best_pos]=bot
        update_board(board)
        if win(bot, board):
            print_win_or_tie("BOT---WIN")

def minmax_helper(max,user,bot,board):
    if max:
        max_score = -1000
        best_move = 8
        for i in range(9):
            if board[i] == " ":
                board[i] = bot
                score = minmax(False, user,bot,board)
                board[i] = " "
                if score > max_score:
                    max_score = score
                    best_move = i
        return best_move
            

def minmax(maximizing, user, bot, board):
    if win(bot, board):
        return 1
    elif win(user, board):
        return -1
    elif tie(board):
        return 0

    if maximizing: 
        max_value = -1000
        for i in range(9):
            if board[i] == " ":
                board[i]=bot
                value = minmax(False, user, bot, board)
                board[i]=" "
                if value > max_value:
                    max_value = value
        return max_value
    else:
        min_value = 1000
        for i in range(9):
            if board[i] == " ":
                board[i]=user
                value = minmax(True, user, bot, board)
                board[i]=" "
                if value < min_value:
                    min_value = value
        return min_value

def tie(board):
    for i in range(9):
        if board[i] == " ":
            return False
    return True

def update_board(board):
    for i in range(9):
        buttons[i].set_text(board[i])

def win(player, board):
    for i in [0, 3, 6]:
        if (
            board[i] == board[i + 1]
            and board[i] == board[i + 2]
            and board[i] == player
        ):
            return True
    for i in [0, 1, 2]:
        if (
            board[i] == board[i + 3]
            and board[i] == board[i + 6]
            and board[i] == player
        ):
            return True
    if (
        board[0] == board[4]
        and board[0] == board[8]
        and board[0] == player
    ):
        return True
    if (
        board[2] == board[4]
        and board[2] == board[6]
        and board[2] == player
    ):
        return True
    return False

def print_win_or_tie(string):
    i=0
    time.sleep(0.5)
    for b in buttons:
        b.set_text(string[i])
        i += 1
    time.sleep(0.3)
    exit(0)
    

if __name__ == "__main__":
    window = Tk()
    window.title("TIC TAC TOE")

    buttons = []
    for i in range(3):
        for j in range(3):
            btn = Button_class(window, i, j, command_fun=start_bot, size=40)
            btn.set_height_width(2, 4)
            buttons.append(btn)

    window.mainloop()

