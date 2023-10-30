import openai
import os
import tiktoken
from dotenv import load_dotenv
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def read_file_and_create_chunks(size):
    tt_encode = tiktoken.get_encoding("cl100k_base")
    with open("input.txt", "r", encoding="utf8") as f:
        text = f.read()
    tokens = tt_encode.encode(text)
    chunks = []
    len_tokens = len(tokens)
    for i in range(0, len_tokens, size):
        chunks.append(tokens[i : i + size])
    return chunks


def create_chunks():
    while True:
        chunks = read_file_and_create_chunks(2500)
        if len(chunks) <= 6:
            print("Chunking completed\n\n")
            break
        print(f"Created {len(chunks)} chunks")
        output_size = 1000
        output_str = ""
        tt_encode = tiktoken.get_encoding("cl100k_base")
        for index, chunk in enumerate(chunks):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {
                        "role": "user",
                        "content": f"summerize in {output_size} words: {tt_encode.decode(chunk)}",
                    }
                ],
                temperature=0.0,
                max_tokens=output_size,
            )
            output_str += response["choices"][0]["message"]["content"]
            output_str += "\n\n"
            print(f"Chunk {index + 1} / {len(chunks)} completed")
        with open("input.txt", "w", encoding="utf8") as f:
            f.write(output_str)


def prompt(whatToDo, howToDo, text):
    # create prompt
    prompt = f"{whatToDo}; {howToDo}; context: {text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"]

def generate_data():
    with open("input.txt", "r", encoding="utf8") as f:
        input_data = f.read()

    detailed_explanation = prompt(
        "provide a detailed explanation in 750 words",
        "tone:formal; style:Expository,Descriptive; Indented Audience:undergraduate students;",
        input_data,
    )
    print("Made detailed explanation")

    informal_explanation = prompt(
        "provide a informal explantion in 1000 words",
        "tone:informal; style:Narrative,Expository,Descriptive; Indented Audience:high-school Students;",
        detailed_explanation,
    )
    print("Made informal explanation")

    with open("data/detailed_explanation.txt", "w", encoding="utf8") as f:
        f.write(detailed_explanation)
    with open("data/informal_explanation.txt", "w", encoding="utf8") as f:
        f.write(informal_explanation)

    print("All API calls completed & data is saved\n\n")


def collect_data_and_generate_report():
    detailed_explanation = open(
        "data/detailed_explanation.txt", "r", encoding="utf8"
    ).read()
    informal_explanation = open(
        "data/informal_explanation.txt", "r", encoding="utf8"
    ).read()

    print("Generating output file: ", end="")
    with open("output.txt", "w", encoding="utf8") as f:
        f.write(f"Informal Explanation:\n\n{informal_explanation}\n\n")
        f.write(f"Detailed Explanation:\n\n{detailed_explanation}\n\n")
    print("Done")


if __name__ == "__main__":
    start_time = time.time()
    create_chunks()
    generate_data()
    collect_data_and_generate_report()
    end_time = time.time()
    
    print(f"\nRun time: {round(end_time - start_time)} seconds")

