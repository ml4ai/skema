from pathlib import Path
from skema.img2mml.api import get_mathml_from_file


def test_get_mathml():
    cwd = Path(__file__).parents[0]
    mathml = get_mathml_from_file(cwd / "data/261.png")
    with open(cwd / "data/261_output.txt") as f:
        output = f.read().strip()

    print(mathml)
    print(output)
    assert mathml == output
test_get_mathml()
