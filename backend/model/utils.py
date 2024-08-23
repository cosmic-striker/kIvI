import os
import json

def load_json(file_path):
    """Loads a JSON file and returns its content."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    """Saves a dictionary as a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def ensure_directory(directory):
    """Ensures that a directory exists. If not, creates it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_message(message):
    """Logs a message to the console or a log file (can be expanded)."""
    print(message)
