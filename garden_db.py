import json
import os

FILE = "gardens.json"


def load_data():
    if not os.path.exists(FILE):
        return {}

    with open(FILE, "r") as f:
        return json.load(f)


def save_data(data):
    with open(FILE, "w") as f:
        json.dump(data, f, indent=4)


def save_garden_info(username, info):

    data = load_data()

    if username not in data:
        data[username] = {"profile": {}, "history": []}

    data[username]["profile"] = info

    save_data(data)


def add_diagnosis(username, entry):

    data = load_data()

    if username not in data:
        data[username] = {"profile": {}, "history": []}

    data[username]["history"].append(entry)

    save_data(data)


def get_garden(username):

    data = load_data()

    return data.get(username, {"profile": {}, "history": []})
