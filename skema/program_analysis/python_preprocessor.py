import subprocess
import tempfile
from pathlib import Path

def preprocess(source: str):
    return convert_python2_to_python3(source)

def convert_python2_to_python3(source: str):
    # Create a temporary file to hold the Python 2 code
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.py') as temp_file:
        temp_file_path = Path(temp_file.name)
        temp_file.write(source)

    # Run 2to3 on the temporary file
    try:
        subprocess.run(['2to3', '--write', '--nobackups', str(temp_file_path)], check=True)
        preprocessed_code = temp_file_path.read_text()
    except subprocess.CalledProcessError:
        preprocessed_code = source
    
    # Clean up the temporary file
    temp_file_path.unlink()

    return preprocessed_code
