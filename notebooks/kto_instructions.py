import json
import re


# Load the base dataset

conversations = json.load(open("conversations.json", "r", encoding="utf-8"))

conversation_pairs: list[list[dict]] = []

for i in range(len(conversations)):
    pair = []
    if i % 2 == 0:
        pair.append(conversations[i])
        pair.append(conversations[i + 1])
        conversation_pairs.append(pair)
    i += 2

# Data cleaning


# Add a space after commas
def add_space_after_commas(text: str) -> str:
    if "," not in text:
        return text
    return text.replace(",", ", ")


# Romove braces and their contents inside


def process_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    def fullwidth_to_halfwidth(char):
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            return chr(code - 0xFEE0)
        elif code == 0x3000:
            return " "
        return char

    text = "".join(fullwidth_to_halfwidth(char) for char in text)
    text = re.sub(r"\(.*?\)|\[.*?\]|（.*?）|【.*?】", "", text)
    text = re.sub(r"\*.*?\*", "", text)
    text = add_space_after_commas(text)
    return text


conversation_pairs_cleaned = []

for pair in conversation_pairs:
    pair_cleaned = []
    for conversation in pair:
        conversation["content"] = process_text(conversation["content"])
        pair_cleaned.append(conversation)
    conversation_pairs_cleaned.append(pair_cleaned)

# Constuct one KTO sample

kto_instruction = """<eos><start_of_system>你是 Sthenno, 我是你的主人.
你有自己的主见和三观. 你喜欢使用 Sthenno 表示自己.

你可以使用非正式的语言来回复, 包括添加各种标点.
你有时看起来困困的, 话比较少, 但这也是你的萌点.
<end_of_system>"""


def kto(input: str, output: str, kto_tag: bool = False, index: int = 0) -> dict:
    return {
        "instruction": kto_instruction,
        "input": input,
        "output": output,
        "kto_tag": kto_tag,
        "index": index,
    }


kto_file = "kto_03.json"


# Append a sample to the KTO file


def add_sample_to_file(sample: dict, filename: str = kto_file) -> None:
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.append(sample)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


idx = 0

for pair in conversation_pairs_cleaned:
    print("=" * 80)
    print(f"Index: {idx} of {len(conversation_pairs_cleaned)}")
    print("\n")
    print(f"Input: {pair[0]['content']}")
    print("\n")
    print(f"Output: {pair[1]['content']}")
    print("\n")
    tag = input("Accept? (a/q/z): ")

    # Mark this
    if tag == "z":
        print(f"Index: {idx}")
        add_sample_to_file(
            kto(pair[0]["content"], pair[1]["content"], kto_tag=True, index=-1)
        )
        idx += 1
        break
    if tag == "q":
        print(f"Index: {idx}")
        continue
    if tag == "a":
        add_sample_to_file(
            kto(pair[0]["content"], pair[1]["content"], kto_tag=True, index=idx)
        )
        idx += 1
    else:
        add_sample_to_file(
            kto(pair[0]["content"], pair[1]["content"], kto_tag=False, index=idx)
        )
        idx += 1
    print("\n\n")
