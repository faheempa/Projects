import pandas as pd

# get data from excel file
data = pd.read_excel("data.xlsx", sheet_name="TIME_TABLE")

# get column names
columns = data.columns.tolist()
columns = [col for col in columns if not col.startswith("Unnamed")]

# map day to index
day_to_index = {}
col_index = 1
for day in columns[1:]:
    day_to_index[day] = col_index
    col_index += 1

# get name of teachers
staffs = pd.read_excel("data.xlsx", sheet_name="NAME_OF_TEACHERS")
staffs = staffs[staffs["CODE"].notna()]
code_to_name = {}
for code, name in zip(staffs["CODE"], staffs["NAME"]):
    code_to_name[code] = name

# datastructure to new table
output = {}
for name in code_to_name.values():
    if name == "nan":
        continue
    output[name] = ["" for _ in range(31)]
    output[name][0] = name


# function to split
def split_by_slash_and_comma(string):
    strs = string.replace("/", ",").replace(",", " ").split(" ")
    strs = [s.strip() for s in strs]
    return strs


def split_string_by_hyphen(string):
    strs = string.split("-")
    if len(strs) == 1:
        return strs[0], ""
    return strs[0], strs[1]


def split_cell(string):
    subject, teachers = split_string_by_hyphen(string)
    teachers = split_by_slash_and_comma(teachers)
    subject = subject.strip()
    return subject, teachers

# loop through each row
for index, row in data.iterrows():
    try: 
        class_name = row["CLASSES"]
        for col in columns[1:]:
            if data[col][index] == "nan":
                continue
            subjects, teachers = split_cell(data[col][index])
            for teacher in teachers:
                if teacher not in code_to_name:
                    continue
                if output[code_to_name[teacher]][day_to_index[col]] != "":
                    output[code_to_name[teacher]][day_to_index[col]] += " + "
                output[code_to_name[teacher]][day_to_index[col]] += f"{class_name} - {subjects}"
    except Exception as e:
        print("Error at column: ", col, " row: ", index, "value: ", data[col][index])   

# make a dataframe with 31 columns
df = pd.DataFrame(columns=["Name", *columns[1:]], data=output.values())
df.to_excel("output.xlsx", index=False)

print("Success")