from sql_functions import *
from tkinter import ttk
from tkinter import *
from tkinter import messagebox


def rent_check_fun():
    try:
        sid = student_id_rent_term.get().upper()
        bid = book_id_rent_term.get()
        if bid == "" and sid == "":
            messagebox.showerror(
                title="Incomplete details", message="Enter complete details"
            )
            return
        name = select_from_table(
            cur,
            student,
            select="NAME",
            id=False,
            condition=f"KTU_ID = '{sid}'",
            rtn=True,
        )
        if name == []:
            messagebox.showerror(
                title="Incorrect ID", message="No such Student ID exist"
            )
            return
        book_details = select_from_table(
            cur,
            book,
            select="Book_name, Available",
            condition=f"Book_ID = {int(bid)}",
            id=False,
            rtn=True,
        )
        if book_details == []:
            messagebox.showerror(title="Incorrect ID", message="No such Book ID exist")
            return
        student_name_rent_term.config(text=name[0][0])
        book_name_rent_term.config(text=book_details[0][0])
        avail_rent_term.config(text=book_details[0][1])
    except:
        pass


def rent_confirm_fun():
    try:
        bid = book_id_rent_term.get()
        sid = student_id_rent_term.get().upper()
        if bid == "" and sid == "":
            messagebox.showerror(
                title="Incomplete details", message="Enter complete details"
            )
            return
        book_avail = (
            select_from_table(
                cur,
                book,
                condition=f"Book_ID = {int(bid)} and Available = 'yes'",
                id=False,
                rtn=True,
            )
            != []
        )
        if book_avail == False:
            messagebox.showinfo(
                title="Not Available", message="Selected book is not available"
            )
            return
        update_records(
            cur, book, set="Available = 'no'", condition=f"Book_ID = {int(bid)}"
        )
        rent_clear_fun()
        insert_into_table(
            cur, rented, value_list=[(int(bid), sid, get_current_time(), " ")]
        )
        print_table(cur, tv_1, book, bookp)
        print_table(cur, tv_3, rented, rentedp, order="d")
    except Exception as e:
        print(e)


def rent_clear_fun():
    book_id_rent_term.delete(0, END)
    student_id_rent_term.delete(0, END)
    book_name_rent_term.config(text="")
    student_name_rent_term.config(text="")
    avail_rent_term.config(text="")


def get_current_time():
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%b-%d-%Y  %I:%M %p")


def clear_tab_0():
    rent_clear_fun()
    book_id_return_term.delete(0, END)


def return_fun():
    if book_id_return_term.get().isdecimal() == False:
        messagebox.showerror(title="Incorrect ID", message="Enter Book ID")
        book_id_return_term.delete(0, END)
        return
    bid = int(book_id_return_term.get())
    if select_from_table(cur, book, condition=f"Book_ID = {bid}", rtn=True) == []:
        messagebox.showerror(title="Incorrect ID", message="No such Book ID exist")
        book_id_return_term.delete(0, END)
        return
    if (
        select_from_table(
            cur, book, condition=f"Book_ID = {bid} and Available='no'", rtn=True
        )
        == []
    ):
        messagebox.showerror(title="Not Rented", message="This book was not rented")
        book_id_return_term.delete(0, END)
        return
    ctime = get_current_time()
    update_records(
        cur, rented, set=f"return_time = '{ctime}'", condition=f"Book_ID = {bid}"
    )
    update_records(cur, book, set="Available = 'yes'", condition=f"Book_ID = {bid}")
    print_table(cur, tv_1, book, bookp)
    print_table(cur, tv_3, rented, rentedp, order="d")
    book_id_return_term.delete(0, END)


def print_table(cur, tv, table_name, sort=None, order="asc"):
    tv.delete(*tv.get_children())
    data = select_from_table(
        cur, table_name, rtn=True, id=False, sort=sort, order=order
    )
    for d in data:
        tv.insert("", "end", values=d)


def tv_1_fun(e):
    try:
        item = tv_1.item(tv_1.focus())
        book_name_term_tv_1.delete(0, END)
        book_name_term_tv_1.insert(0, item["values"][1])
        author_term_tv_1.delete(0, END)
        author_term_tv_1.insert(0, item["values"][2])
    except:
        pass


def search_fun_tv_1():
    term = search_term_tv_1.get()
    if term == "yes" or term == "no":
        cur.execute(
            f"select * from {book} where Available = '{term}' order by Book_name"
        )
    elif term.isdecimal():
        cur.execute(
            f"select * from {book} where Book_ID = {int(term)} order by Book_name"
        )
    else:
        cur.execute(
            f"select * from {book} where Book_name like '%{term}%' or Author_name like '%{term}%' order by Book_name"
        )
    data = cur.fetchall()
    tv_1.delete(*tv_1.get_children())
    for d in data:
        tv_1.insert("", "end", values=d)


def clear_fun_search_tv_1():
    print_table(cur, tv_1, book, bookp)
    search_term_tv_1.delete(0, END)


def update_tv_1():
    try:
        item = tv_1.item(tv_1.focus())
        bid, name, author, avail = (
            item["values"][0],
            book_name_term_tv_1.get(),
            author_term_tv_1.get(),
            item["values"][3],
        )
        if name == "" or author == "":
            messagebox.showinfo(
                title="incomplete details", message="Enter the details of the book"
            )
            return
        delete_records(cur, book, condition=f"Book_ID = {item['values'][0]}")
        value = [(bid, name, author, avail)]
        insert_into_table(cur, book, value_list=value)
        clear_fun_data_tv_1()
        print_table(cur, tv_1, book, bookp)
    except:
        pass


def clear_fun_data_tv_1():
    book_name_term_tv_1.delete(0, END)
    author_term_tv_1.delete(0, END)


def delete_fun_data_tv_1():
    try:
        item = tv_1.item(tv_1.focus())
        delete_records(cur, book, condition=f"Book_ID = {item['values'][0]}")
        clear_fun_data_tv_1()
        print_table(cur, tv_1, book, bookp)
    except:
        pass


def add_fun_data_tv_1():
    try:
        global bid
        name, author, avail = (book_name_term_tv_1.get(), author_term_tv_1.get(), "yes")
        if name == "" or author == "":
            messagebox.showinfo(
                title="incomplete details", message="Enter the details of the book"
            )
            return
        value = [(bid, name, author, avail)]
        insert_into_table(cur, book, value_list=value)
        print_table(cur, tv_1, book, bookp)
        clear_fun_data_tv_1()
        clear_fun_search_tv_1()
        bid += 1
    except:
        pass


def tv_2_fun(e):
    try:
        item = tv_2.item(tv_2.focus())
        ktu_id_term_tv_2.delete(0, END)
        ktu_id_term_tv_2.insert(0, item["values"][0])
        student_name_term_tv_2.delete(0, END)
        student_name_term_tv_2.insert(0, item["values"][1])
    except:
        pass


def search_fun_tv_2():
    term = search_term_tv_2.get()
    cur.execute(
        f"select * from {student} where KTU_ID like '%{term}%' or NAME like '%{term}%' order by NAME"
    )
    data = cur.fetchall()
    tv_2.delete(*tv_2.get_children())
    for d in data:
        tv_2.insert("", "end", values=d)


def clear_fun_search_tv_2():
    print_table(cur, tv_2, student, studentp)
    search_term_tv_2.delete(0, END)


def update_tv_2():
    try:
        item = tv_2.item(tv_2.focus())
        sid, name = (ktu_id_term_tv_2.get().upper(), student_name_term_tv_2.get())
        if name == "" or sid == "":
            messagebox.showinfo(
                title="incomplete details", message="Enter the details of the student"
            )
            return
        delete_records(cur, student, condition=f"KTU_ID = '{item['values'][0]}'")
        value = [(sid, name)]
        insert_into_table(cur, student, value_list=value)
        clear_fun_data_tv_2()
        print_table(cur, tv_2, student, studentp)
    except Exception as e:
        print(e)


def clear_fun_data_tv_2():
    ktu_id_term_tv_2.delete(0, END)
    student_name_term_tv_2.delete(0, END)


def delete_fun_data_tv_2():
    try:
        item = tv_2.item(tv_2.focus())
        delete_records(cur, student, condition=f"KTU_ID = '{item['values'][0]}'")
        clear_fun_data_tv_2()
        print_table(cur, tv_2, student, studentp)
    except:
        pass


def add_fun_data_tv_2():
    try:
        kid, name = (ktu_id_term_tv_2.get().upper(), student_name_term_tv_2.get())
        if name == "" or kid == "":
            messagebox.showinfo(
                title="incomplete details", message="Enter the details of the student"
            )
            return
        value = [(kid, name)]
        insert_into_table(cur, student, value_list=value)
        print_table(cur, tv_2, student, studentp)
        clear_fun_data_tv_2()
        clear_fun_search_tv_2()
    except:
        pass


if __name__ == "__main__":
    book = "books"
    bookp = "Book_name"
    student = "students"
    studentp = "KTU_ID"
    rented = "books_rented"
    rentedp = "rowid"
    db = create_db("database")
    cur = get_cur(db)
    window = Tk()
    window.title("LIBRARY MANAGEMENT SYSTEM")
    window.geometry("950x850")
    window.config(background="white")

    try:
        notebook = ttk.Notebook(window)
        tab_0 = Frame(notebook)
        tab_1 = Frame(notebook)
        tab_2 = Frame(notebook)
        tab_3 = Frame(notebook)
        notebook.add(tab_0, text="Management")
        notebook.add(tab_1, text="Book Details")
        notebook.add(tab_2, text="Student Details")
        notebook.add(tab_3, text="History")
        notebook.pack(expand=True, fill="both")

        # management
        wrapper1 = LabelFrame(tab_0, text=" RENT ", font=("arial", 15))
        wrapper2 = LabelFrame(tab_0, text=" RETURN ", font=("arial", 15))
        wrapper1.pack(fill="both", padx=20, pady=10)
        wrapper2.pack(fill="both", padx=20, pady=10)

        rent_wrapper_tab_0 = LabelFrame(wrapper1, border=False, font=("arial", 15))
        rent_wrapper_tab_0.pack(pady=20)
        book_id_rent = Label(
            rent_wrapper_tab_0,
            text="BOOK ID",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        book_id_rent.grid(row=1, column=0)
        book_id_rent_term = Entry(
            rent_wrapper_tab_0, font=("arial", 15), fg="white", bg="black", width=50
        )
        book_id_rent_term.grid(row=1, column=1)
        student_id_rent = Label(
            rent_wrapper_tab_0,
            text="STUDENT ID",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        student_id_rent.grid(row=2, column=0)
        student_id_rent_term = Entry(
            rent_wrapper_tab_0, font=("arial", 15), fg="white", bg="black", width=50
        )
        student_id_rent_term.grid(row=2, column=1)

        check_btn_rent = Button(
            rent_wrapper_tab_0,
            text="Check",
            command=rent_check_fun,
            font=("arial", 15),
            width=10,
        )
        check_btn_rent.grid(row=3, column=0, columnspan=2, pady=10)

        student_name_rent = Label(
            rent_wrapper_tab_0,
            text="STUDENT NAME",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        student_name_rent.grid(row=4, column=0)
        student_name_rent_term = Label(
            rent_wrapper_tab_0,
            text="",
            font=("arial", 15),
            fg="white",
            bg="black",
            width=50,
        )
        student_name_rent_term.grid(row=4, column=1)
        book_name_rent = Label(
            rent_wrapper_tab_0,
            text="BOOK NAME",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        book_name_rent.grid(row=5, column=0)
        book_name_rent_term = Label(
            rent_wrapper_tab_0,
            text="",
            font=("arial", 15),
            fg="white",
            bg="black",
            width=50,
        )
        book_name_rent_term.grid(row=5, column=1)
        avail_rent = Label(
            rent_wrapper_tab_0,
            text="BOOK AVAILABLE",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        avail_rent.grid(row=6, column=0)
        avail_rent_term = Label(
            rent_wrapper_tab_0,
            text="",
            font=("arial", 15),
            fg="white",
            bg="black",
            width=50,
        )
        avail_rent_term.grid(row=6, column=1)
        confirm_btn_rent = Button(
            rent_wrapper_tab_0,
            text="Confirm",
            command=rent_confirm_fun,
            font=("arial", 15),
            width=10,
        )
        confirm_btn_rent.grid(row=7, column=0, columnspan=2, pady=10)

        return_wrapper_tab_0 = LabelFrame(wrapper2, border=False, font=("arial", 15))
        return_wrapper_tab_0.pack(pady=20)
        book_id_return = Label(
            return_wrapper_tab_0,
            text="BOOK ID",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        book_id_return.grid(row=1, column=0)
        book_id_return_term = Entry(
            return_wrapper_tab_0, font=("arial", 15), fg="white", bg="black", width=50
        )
        book_id_return_term.grid(row=1, column=1)
        check_btn_return = Button(
            return_wrapper_tab_0,
            text="Return",
            command=return_fun,
            font=("arial", 15),
            width=10,
        )
        check_btn_return.grid(row=2, column=0, columnspan=2, pady=10)

        clear_btn_rent = Button(
            tab_0,
            text="Clear All",
            command=clear_tab_0,
            font=("arial", 15),
            width=10,
        )
        clear_btn_rent.pack(pady=50)

        # book details tab
        tv_1 = ttk.Treeview(tab_1, columns=(1, 2, 3, 4), show="headings", height=20)
        tv_1.pack(pady=20, fill="both", padx=20)
        tv_1.bind("<Double 1>", tv_1_fun)
        tv_1.heading(1, text="BOOK_ID")
        tv_1.column(1, width=100)
        tv_1.heading(2, text="BOOK_NAME")
        tv_1.column(2, width=350)
        tv_1.heading(3, text="AUTHOR_NAME")
        tv_1.column(3, width=350)
        tv_1.heading(4, text="AVAILABILE")
        tv_1.column(4, width=100)
        print_table(cur, tv_1, book, bookp)

        search_wrapper_tab_1 = LabelFrame(
            tab_1, border=False, text="Search: ", font=("arial", 15)
        )
        search_wrapper_tab_1.pack(pady=40)
        search_term_tv_1 = Entry(
            search_wrapper_tab_1, font=("arial", 15), fg="white", bg="black", width=45
        )
        search_term_tv_1.pack(side=LEFT)
        search_btn_tv_1 = Button(
            search_wrapper_tab_1,
            text="Search",
            font=("arial", 15),
            command=search_fun_tv_1,
            width=10,
        )
        search_btn_tv_1.pack(side=LEFT)
        clear_btn_tv_1 = Button(
            search_wrapper_tab_1,
            text="clear",
            command=clear_fun_search_tv_1,
            font=("arial", 15),
            width=10,
        )
        clear_btn_tv_1.pack()

        data_wrapper_tab_1 = LabelFrame(
            tab_1, border=False, text="Book details: ", font=("arial", 15)
        )
        data_wrapper_tab_1.pack(pady=20)
        book_name_tv_1 = Label(
            data_wrapper_tab_1,
            text="Book Name",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        book_name_tv_1.grid(row=1, column=0)
        book_name_term_tv_1 = Entry(
            data_wrapper_tab_1, font=("arial", 15), fg="white", bg="black", width=50
        )
        book_name_term_tv_1.grid(row=1, column=1)
        author_tv_1 = Label(
            data_wrapper_tab_1,
            text="Author Name",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        author_tv_1.grid(row=2, column=0)
        author_term_tv_1 = Entry(
            data_wrapper_tab_1, font=("arial", 15), fg="white", bg="black", width=50
        )
        author_term_tv_1.grid(row=2, column=1)

        btn_wrapper_tab_1 = LabelFrame(tab_1, border=False)
        btn_wrapper_tab_1.pack()
        update_btn_tv_1 = Button(
            btn_wrapper_tab_1,
            text="Update",
            command=update_tv_1,
            font=("arial", 15),
            width=10,
        )
        update_btn_tv_1.pack(side=LEFT)
        clear_btn_data_tv_1 = Button(
            btn_wrapper_tab_1,
            text="Clear",
            command=clear_fun_data_tv_1,
            font=("arial", 15),
            width=10,
        )
        clear_btn_data_tv_1.pack(side=LEFT)
        delete_btn_data_tv_1 = Button(
            btn_wrapper_tab_1,
            text="Delete",
            command=delete_fun_data_tv_1,
            font=("arial", 15),
            width=10,
        )
        delete_btn_data_tv_1.pack(side=LEFT)
        cur.execute(f"select max(Book_ID) from {book}")
        bid = int(cur.fetchall()[0][0]) + 1
        add_btn_data_tv_1 = Button(
            btn_wrapper_tab_1,
            text="Add",
            command=add_fun_data_tv_1,
            font=("arial", 15),
            width=10,
        )
        add_btn_data_tv_1.pack(side=LEFT)

        # student details tab
        tv_2 = ttk.Treeview(tab_2, columns=(1, 2), show="headings", height=20)
        tv_2.pack(pady=20, fill="both", padx=20)
        tv_2.bind("<Double 1>", tv_2_fun)
        tv_2.heading(1, text="KTU_ID")
        tv_2.column(1, width=250)
        tv_2.heading(2, text="NAME")
        tv_2.column(2, width=650)
        print_table(cur, tv_2, student, studentp)

        search_wrapper_tab_2 = LabelFrame(
            tab_2, border=False, text="Search: ", font=("arial", 15)
        )
        search_wrapper_tab_2.pack(pady=40)
        search_term_tv_2 = Entry(
            search_wrapper_tab_2, font=("arial", 15), fg="white", bg="black", width=45
        )
        search_term_tv_2.pack(side=LEFT)
        search_btn_tv_2 = Button(
            search_wrapper_tab_2,
            text="Search",
            font=("arial", 15),
            command=search_fun_tv_2,
            width=10,
        )
        search_btn_tv_2.pack(side=LEFT)
        clear_btn_tv_2 = Button(
            search_wrapper_tab_2,
            text="clear",
            command=clear_fun_search_tv_2,
            font=("arial", 15),
            width=10,
        )
        clear_btn_tv_2.pack()

        data_wrapper_tab_2 = LabelFrame(
            tab_2, border=False, text="Student details: ", font=("arial", 15)
        )
        data_wrapper_tab_2.pack(pady=20)
        ktu_id_tv_2 = Label(
            data_wrapper_tab_2,
            text="KTU-ID",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        ktu_id_tv_2.grid(row=1, column=0)
        ktu_id_term_tv_2 = Entry(
            data_wrapper_tab_2, font=("arial", 15), fg="white", bg="black", width=50
        )
        ktu_id_term_tv_2.grid(row=1, column=1)
        student_name_tv_2 = Label(
            data_wrapper_tab_2,
            text="Name",
            font=("arial", 15),
            bg="white",
            fg="black",
            width=20,
        )
        student_name_tv_2.grid(row=2, column=0)
        student_name_term_tv_2 = Entry(
            data_wrapper_tab_2, font=("arial", 15), fg="white", bg="black", width=50
        )
        student_name_term_tv_2.grid(row=2, column=1)

        btn_wrapper_tab_2 = LabelFrame(tab_2, border=False)
        btn_wrapper_tab_2.pack()
        update_btn_tv_2 = Button(
            btn_wrapper_tab_2,
            text="Update",
            command=update_tv_2,
            font=("arial", 15),
            width=10,
        )
        update_btn_tv_2.pack(side=LEFT)
        clear_btn_data_tv_2 = Button(
            btn_wrapper_tab_2,
            text="Clear",
            command=clear_fun_data_tv_2,
            font=("arial", 15),
            width=10,
        )
        clear_btn_data_tv_2.pack(side=LEFT)
        delete_btn_data_tv_2 = Button(
            btn_wrapper_tab_2,
            text="Delete",
            command=delete_fun_data_tv_2,
            font=("arial", 15),
            width=10,
        )
        delete_btn_data_tv_2.pack(side=LEFT)
        add_btn_data_tv_2 = Button(
            btn_wrapper_tab_2,
            text="Add",
            command=add_fun_data_tv_2,
            font=("arial", 15),
            width=10,
        )
        add_btn_data_tv_2.pack(side=LEFT)

        # history
        tv_3 = ttk.Treeview(tab_3, columns=(1, 2, 3, 4), show="headings")
        tv_3.pack(expand=True, fill="both", padx=20, pady=20)
        tv_3.heading(1, text="Book ID")
        tv_3.column(1, width=200)
        tv_3.heading(2, text="KTU ID")
        tv_3.column(2, width=200)
        tv_3.heading(3, text="RENT_TIME")
        tv_3.column(3, width=250)
        tv_3.heading(4, text="RETURN_TIME")
        tv_3.column(4, width=250)
        print_table(cur, tv_3, rented, rentedp, order="d")

        window.mainloop()

    except Exception as e:
        print(e)
    finally:
        db.commit()
        db.close()
