import json
import os
import bcrypt

USER_FILE = "users.json"


def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    try:
        with open(USER_FILE, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else {}
    except json.JSONDecodeError:
        return {}


def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)


def register_user(username, password):

    username = (username or "").strip()

    if not username or not password:
        raise ValueError("Username and password cannot be empty.")

    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    users = load_users()

    if username in users:
        return False

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    users[username] = hashed

    save_users(users)

    return True


def login_user(username, password):

    users = load_users()

    if username in users:
        return bcrypt.checkpw(password.encode(), users[username].encode())

    return False
