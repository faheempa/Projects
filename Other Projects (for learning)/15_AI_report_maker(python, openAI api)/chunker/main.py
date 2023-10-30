import openai
import os
import tiktoken
from dotenv import load_dotenv
import time

load_dotenv()
openai.api_key = "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"


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
        chunks = read_file_and_create_chunks(2000)
        if len(chunks) <= 5:
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


if __name__ == "__main__":
    create_chunks()
    

