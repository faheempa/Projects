import sqlite3

def create_db(db_name):
    return sqlite3.connect(f"{db_name}.db")

def get_cur(db):
    return db.cursor()

def create_table(cur,table_name, attributes):
    cur.execute(f"create table {table_name} ( {attributes} )")

def insert_into_table(cur, table_name, value_list):
    # value_list is a list of tuples
    for values in value_list:
        value_to_insert = ""
        for x in values:
            if type(x) != str:
                value_to_insert += str(x) + ","
            else:
                value_to_insert +=  f'"{x}",'
        value_to_insert = value_to_insert.rstrip(",")
        cur.execute(f"insert into {table_name} values({value_to_insert})")

def select_from_table(cur, table_name, condition=None, sort=None, id=True, order="asc", rtn=False, select=None):
    if select==None:
        if sort==None:
            if condition==None:
                if id==False:
                    cur.execute(f"select * from {table_name}")
                else:
                    cur.execute(f"select rowid, * from {table_name}")
            else:
                if id==False:
                    cur.execute(f"select * from {table_name} where {condition}")
                else:
                    cur.execute(f"select rowid, * from {table_name} where {condition}")
        else:
            if order=="d":
                order="desc"
            if condition==None:
                if id==False:
                    cur.execute(f"select * from {table_name} order by {sort} {order}")
                else:
                    cur.execute(f"select rowid, * from {table_name} order by {sort} {order}")
            else:
                if id==False:
                    cur.execute(f"select * from {table_name} where {condition} order by {sort} {order}")
                else:
                    cur.execute(f"select rowid, * from {table_name} where {condition} order by {sort} {order}")
    else:
        if sort==None:
            if condition==None:
                if id==False:
                    cur.execute(f"select {select} from {table_name}")
                else:
                    cur.execute(f"select rowid, {select} from {table_name}")
            else:
                if id==False:
                    cur.execute(f"select {select} from {table_name} where {condition}")
                else:
                    cur.execute(f"select rowid, {select} from {table_name} where {condition}")
        else:
            if order=="d":
                order="desc"
            if condition==None:
                if id==False:
                    cur.execute(f"select {select} from {table_name} order by {sort} {order}")
                else:
                    cur.execute(f"select rowid, {select} from {table_name} order by {sort} {order}")
            else:
                if id==False:
                    cur.execute(f"select {select} from {table_name} where {condition} order by {sort} {order}")
                else:
                    cur.execute(f"select rowid, {select} from {table_name} where {condition} order by {sort} {order}")

    if rtn:
        return cur.fetchall()
    for x in cur.fetchall():
        for y in x:
            print(y,"\t",end="")
        print()

def drop_table(cur,table_name):
    cur.execute(f"drop table {table_name}")

def update_records(cur, table_name, set, condition=None):
    if condition==None:
        cur.execute(f"update {table_name} set {set}")
    else:
        cur.execute(f"update {table_name} set {set} where {condition}")
        
def delete_records(cur,table_name,condition):
    cur.execute(f"delete from {table_name} where {condition}")