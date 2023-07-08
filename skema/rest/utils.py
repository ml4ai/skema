from bs4 import BeautifulSoup, Comment


def clean_mml(mml: str) -> str:
    """Cleans/sterilizes pMML for AMR generation service"""
    soup = BeautifulSoup(mml, "html.parser")
    for comment in soup(text=lambda text: isinstance(text, Comment)):
        comment.extract()
    return str(soup).replace("\n", "")
