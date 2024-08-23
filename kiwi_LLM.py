import os

# Define the project structure
project_structure = {
    "backend": {
        "app.py": "",
        "config.py": "",
        "requirements.txt": "",
        "model": {
            "llama2_model.py": "",
            "utils.py": "",
        },
        "tests": {
            "test_app.py": "",
            "test_model.py": "",
        },
    },
    "frontend": {
        "templates": {
            "base.html": "",
            "index.html": "",
            "chat.html": "",
        },
        "static": {
            "css": {
                "styles.css": "",
            },
            "js": {
                "scripts.js": "",
            },
            "assets": {},  # Empty folder for images, fonts, etc.
        },
        "tests": {
            "test_frontend.py": "",
        },
    },
}

# Function to create directories and files
def create_project_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_project_structure(path, content)
        else:
            with open(path, 'w') as file:
                file.write(content)

# Get the current directory
base_path = os.getcwd()

# Create the project structure
create_project_structure(base_path, project_structure)

print(f"Project structure created at: {base_path}")
