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
                model="gpt-3.5-turbo",
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


def prompt(whatToDo, howToDo, text, model="gpt-3.5-turbo-16k"):
    # create prompt
    prompt = f"{whatToDo}; {howToDo}; context: {text}"
    if model == "gpt-3.5-turbo-16k":
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
    if model == "gpt-4":
        response = openai.ChatCompletion.create(
            model="gpt-4",
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
        "provide a detailed explanation in about 1000 words",
        "tone:formal; style:Expository,Descriptive; Indented Audience:undergraduate students;",
        input_data,
    )
    print("Made detailed explanation")

    quick_overview = prompt(
        "Summerize strictly under 200 words",
        "tone:formal; style:Analytical; Indented Audience:Academics;",
        detailed_explanation,
    )
    print("Made quick overview")

    informal_explanation = prompt(
        "provide a informal explantion in 1000 words",
        "tone:informal; style:Narrative,Expository,Descriptive; Indented Audience:high-school Students;",
        detailed_explanation,
    )
    print("Made informal explanation")

    key_points = prompt(
        "list out 10 key points from this project and explain them in 3-5 lines",
        "tone:formal; style:Descriptive; indented audience:undergraduate students; output_format:bullet_points;",
        detailed_explanation,
        model="gpt-4",
    )
    print("Made key points")

    viva_qna = prompt(
        "make 20 viva questions and detailed answers (3-5 lines) for them from this project",
        "tone:formal; indented audience:undergraduate students; output_format:question_and_answer;",
        detailed_explanation,
        model="gpt-4",
    )
    print("Made viva QnA")

    with open("data/detailed_explanation.txt", "w", encoding="utf8") as f:
        f.write(detailed_explanation)
    with open("data/quick_overview.txt", "w", encoding="utf8") as f:
        f.write(quick_overview)
    with open("data/informal_explanation.txt", "w", encoding="utf8") as f:
        f.write(informal_explanation)
    with open("data/key_points.txt", "w", encoding="utf8") as f:
        f.write(key_points)
    with open("data/viva_qna.txt", "w", encoding="utf8") as f:
        f.write(viva_qna)

    print("All API calls completed & data is saved\n\n")


def collect_data_and_generate_report():
    detailed_explanation = open(
        "data/detailed_explanation.txt", "r", encoding="utf8"
    ).read()
    quick_overview = open("data/quick_overview.txt", "r", encoding="utf8").read()
    informal_explanation = open(
        "data/informal_explanation.txt", "r", encoding="utf8"
    ).read()
    key_points = open("data/key_points.txt", "r", encoding="utf8").read()
    viva_qna = open("data/viva_qna.txt", "r", encoding="utf8").read()

    print("Generating output file: ", end="")
    with open("output.txt", "w", encoding="utf8") as f:
        f.write(f"Quick Overview:\n\n{quick_overview}\n\n")
        f.write(f"Informal Explanation:\n\n{informal_explanation}\n\n")
        f.write(f"Detailed Explanation:\n\n{detailed_explanation}\n\n")
        f.write(f"Key Points:\n\n{key_points}\n\n")
        f.write(f"Viva QNA:\n\n{viva_qna}\n\n")
    print("Done")


if __name__ == "__main__":
    start_time = time.time()
    create_chunks()
    generate_data()
    collect_data_and_generate_report()
    end_time = time.time()

    run_time = end_time - start_time
    minutes = run_time // 60
    seconds = run_time % 60
    print(f"\nRun time: {round(minutes)}min {round(seconds)}sec")

