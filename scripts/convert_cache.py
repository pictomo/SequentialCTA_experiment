# Usage: python convert_cache.py <client_name>

import sys
import os

abspath = os.path.dirname(os.path.abspath(sys.argv[0]))
top_path = os.path.dirname(abspath)
sys.path.append(os.path.join(top_path, "modules"))

import json
import csv


client_name = sys.argv[1]


if __name__ == "__main__":
    with open(
        os.path.join(top_path, "notebooks/haio_cache/1c57d50d996eb63a8be99364e27bc016"),
        "r",
        encoding="utf-8",
    ) as file:
        data_src = json.load(file)  # JSONをPythonのdictに変換

    data = []

    for value in data_src["data_lists"].values():
        for item in value["answer_list"].values():
            if item["client"] == client_name:
                data.append([value["data_list"][0], item["answer"]])

    with open(
        os.path.join(top_path, f"sandbox/{client_name}.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(data)
