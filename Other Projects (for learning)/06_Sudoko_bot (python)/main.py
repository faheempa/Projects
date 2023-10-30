from tkinter import *
from button_class import *

window = Tk()
window.title("SUDOKO")
window.config(background="white")


def start():
    board = [[0 for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            if buttons[i][j].text != " ":
                board[i][j] = int(buttons[i][j].text)
    bot(board)
    for i in range(9):
        for j in range(9):
            buttons[i][j].set_color(bg="light green")


def bot(board):
    for x in range(9):
        for y in range(9):
            if board[x][y] == 0:
                for n in range(1, 10):
                    if possible(x, y, n, board):
                        board[x][y] = n
                        bot(board)
                        board[x][y] = 0
                return
    update_buttons(board)


def possible(x, y, n, board):
    for i in range(9):
        if board[i][y] == n:
            return False
        if board[x][i] == n:
            return False
    for i in [1, 4, 7]:
        for j in range(i - 1, (i + 1) + 1):
            if x == j:
                x = i
            if y == j:
                y = i
    for i in range(x - 1, (x + 1) + 1):
        for j in range(y - 1, (y + 1) + 1):
            if board[i][j] == n:
                return False
    return True


def update_buttons(board):
    for i in range(9):
        for j in range(9):
            a = str(board[i][j])
            buttons[i][j].set_text(a)


def add_text(x, y):
    if entry.get().isdigit():
        if 0 < int(entry.get()) < 10:
            buttons[x][y].set_text(entry.get())
            buttons[x][y].set_color("light green")
            entry.delete(0, END)


buttons = [[None for i in range(9)] for i in range(9)]
for i in range(9):
    for j in range(9):
        btn = Button_class(window, i, j, command_fun=add_text)
        btn.set_height_width(1, 3)
        buttons[i][j] = btn


entry = Entry(window, font=("arial", 20), fg="red", bg="black", width=3, borderwidth=5)
entry.grid(row=10, column=3, columnspan=3)

start_the_bot = Button(window, text="Start the bot", command=start)
start_the_bot.grid(row=11, column=3, columnspan=3)


window.mainloop()
