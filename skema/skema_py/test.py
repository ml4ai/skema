import requests
from pathlib import Path
from skema.utils.fold import dictionary_to_gromet_json, del_nulls
''''''
# Multi file
system = {
  "files": [
    "example1.py",
    "dir/example2.py"
  ],
  "blobs": [
    "greet = lambda: print('howdy!')\ngreet()",
    "#Variable declaration\nx=2\n#Function definition\ndef foo(x):\n    '''Increment the input variable'''\n    return x+1"
  ],
  "system_name": "example-system",
  "root_name": "example-system",
  "comments": {
    "files": {
      "example2.py": {
        "comments": [
          {
            "contents": "Variable declaration",
            "line_number": 0
          },
          {
            "contents": "Function definition",
            "line_number": 2
          }
        ],
        "docstrings": {
          "foo": [
            "Increment the input variable"
          ]
        }
      }
    }
  }
}
response = requests.post("http://localhost:8000/code2fn/fn-given-filepaths", json=system)
gromet_json = response.json()

output_path = Path("output_client.json")
output_path.write_text(dictionary_to_gromet_json(del_nulls(gromet_json)))

print(response.text)