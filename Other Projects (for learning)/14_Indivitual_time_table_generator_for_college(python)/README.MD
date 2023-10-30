# Time table generator

## Description

This is a simple time table generator written in python and uses the pandas library for data manipulation.

## Installation

```bash
pip3 install pandas 
```

## Usage

```python
python main.py
```

## Instructions

Points to note in the input data file:
1. The input data file should be an excel file.
2. The first sheet should be named "TIME_TABLE" and the second sheet should be named "NAME_OF_TEACHERS".
3. The first sheet should have a total of 31 columns
4. The first column should be "CLASSES" and the next 30 columns should be a representation of weekdays * 6 periods.
5. The second sheet should have a total of 2 columns
6. The first column should be "CODE" and the second column should be "NAME"
7. All the data cell should be in this format: "SUBJECT_CODE-TEACHER_CODE"
8. Empty cells should be left blank and will not be considered in the time table generation.
9. Cell which does not follow the above format will be considered as an empty cell.
10. No two teachers should have the same code.
11. No merge cells should be used in the excel file.
12. The output file will be generated in the same directory as the input file.

## Working

The program works by first reading the data from the Excel file into a Pandas DataFrame. Then, it gets the names of the teachers and the column names of the DataFrame. Next, it creates a dictionary to map days to indices. The program then loops through each row in the DataFrame. For each row, it gets the class name and the subjects and teachers that are being taught in that class. It then splits the subjects and teachers into lists. Finally, the program adds the class name and subject to the new table at the appropriate row and column. Once the loop is finished, the program saves the new table to an Excel file. In summary, the program takes an Excel file containing class timetables and generates a new Excel file with a more readable format, where each teacher has their own timetable with the classes they teach and the days they teach them on.

