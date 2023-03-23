from preprocessing.preprocess_mml import simplification
import sys

def main(mml):
    while(True):
        return(simplification(mml))

if __name__ == '__main__':
    mml = sys.argv[1]
    main(mml)
