from pathlib import Path

# Create a temp file to store the program being tested
def create_temp_file(file_content_string, extension):
    temp_file = open(f"temp.{extension}", "w")
    temp_file.write(file_content_string)
    temp_file.close()

# Attempt to delete the temp file, if it exists
def delete_temp_file(extension):
    temp_file_path = Path(f"temp.{extension}")
    if temp_file_path.exists():
        temp_file_path.unlink()
    
# Simple testing
def main():
    create_temp_file("x = 2", "py")
    print(Path("temp.py").exists())
    
    delete_temp_file("py")
    print(Path("temp.py").exists())

main()