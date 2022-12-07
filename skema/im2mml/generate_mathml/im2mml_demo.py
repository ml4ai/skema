import requests

def get_mml(image_path: str) -> str:
    '''
    It sends the http requests to put in an image to translate it into MathML.
    '''
    with open(image_path, 'rb') as f:
        r = requests.put("http://127.0.0.1:8000/get_mml", files = {"file": f})
    return r.text

mml = get_mml("images/sir.png")
print(mml)
