""" Temporary test script for development This will change to unit tests and example usage script """

from skema.utils.fold import dictionary_to_gromet_json, del_nulls
from skema.text_reading.mention_linking.gromet_linker.comment_aligner import (
    CommentAlignerHelper,
    GrometFNModuleCommentAligner,
)
from skema.text_reading.mention_linking.gromet_linker.comment_debugger import (
    CommentDebugger,
)
from skema.text_reading.mention_linking.gromet_linker.gromet_helper import (
    GrometHelper,
)
from skema.text_reading.mention_linking.gromet_linker.source_comments import (
    SourceComments,
)
from skema.text_reading.mention_linking.gromet_linker.text_reading_linker import (
    TextReadingLinker,
)
from skema.text_reading.mention_linking.gromet_linker.time_stamper import (
    DebugTimeStamper,
    NowTimeStamper,
)
from skema.text_reading.mention_linking.gromet_linker.uid_stamper import (
    DocIdStamper,
    UidStamper,
)
from skema.text_reading.mention_linking.gromet_linker.variable_name_matcher import (
    VariableNameMatcher,
)

from skema.utils.module_to_fn_collection import (
    module_to_fn_collection
)

import os
import argparse


class Paths:
    def __init__(
        self,
        base_path: str,
        embeddings_path: str,
        gromet_file: str,
        comment_file: str,
        extraction_file: str,
    ):
        self.embeddings_path = embeddings_path
        self.gromet_path = f"{base_path}/gromet/{gromet_file}"
        self.comments_path = f"{base_path}/comments/{comment_file}"
        self.extractions_path = f"{base_path}/extractions/{extraction_file}"
        self.regression_path = f"{base_path}/regression/{gromet_file}"
        self.test_path = f"{base_path}/test/{gromet_file}"
        assert self.isValid()

    def isValid(self) -> bool:
        for path in [
            self.gromet_path,
            self.comments_path,
            self.extractions_path,
            self.embeddings_path,
            self.embeddings_path + ".vectors.npy",
        ]:
            if not os.path.exists(path):
                print(f"Path {path} does not seem to exist.")
                return False
        return True


class Tester:
    def __init__(self, paths: Paths, debug: bool = True):
        debugger = CommentDebugger.create(debug)
        time_stamper = DebugTimeStamper() if debug else NowTimeStamper()
        uid_stamper = DocIdStamper() if debug else UidStamper()
        variable_name_matcher = VariableNameMatcher(
            "python"
        )  # TODO: Use actual language.
        source_comments = SourceComments.from_file(paths.comments_path)
        linker = TextReadingLinker(
            paths.extractions_path, paths.embeddings_path
        )
        self.paths = paths
        self.gromet_fn_module = GrometHelper.json_to_gromet(
            paths.gromet_path
        ).modules[
            0
        ]  # TODO, make this iterable over all the modules in the gromet
        self.comment_aligner_helper = CommentAlignerHelper(
            debugger,
            time_stamper,
            uid_stamper,
            self.gromet_fn_module,
            variable_name_matcher,
            source_comments,
            linker,
        )

    def test(self):
        comment_aligner = GrometFNModuleCommentAligner(
            self.gromet_fn_module,
            self.comment_aligner_helper,
            self.paths.embeddings_path,
        )
        comment_aligner.align()
        # Save gromet file with the new metadata aligned.
        # with open(self.paths.test_path, "w") as file:
        #     file.write(
        #         dictionary_to_gromet_json(
        #             del_nulls(self.gromet_fn_module.to_dict())
        #         )
        #     )
        print(
            dictionary_to_gromet_json(
                del_nulls(module_to_fn_collection(self.gromet_fn_module).to_dict())
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base_path",
        help="Base path of the directory with the files to align",
    )
    parser.add_argument(
        "-e",
        "--embeddings_path",
        help="Path of the embeddings model used for alignment",
    )
    parser.add_argument("-g", "--gromet_name", help="Gromet FN file name")
    parser.add_argument(
        "-c", "--comments_name", help="Code comments file name"
    )
    parser.add_argument(
        "-x", "--extractions_name", help="TR extractions file name", default=''
    )

    args = parser.parse_args()

    # This places the files just outside of the repo directory.

    base_path = args.base_path  # "../../../../mention_linking_files"

    embeddings_path = args.embeddings_path

    # embeddings_path = (
    # "../../../../word_embeddings/epi+code_comments/embeddings.kv"
    # )
    # base_path = "data"
    # embeddings_path = "/data/covid_comments_models/xdd_covid_19_1x_word2vec/alternate/embeddings.kv"

    paths = Paths(
        base_path,
        embeddings_path,
        args.gromet_name,  # "CHIME_SIR--Gromet-FN-auto.json",
        args.comments_name,
        args.extractions_name,
    )

    # The extraction seems to be incomplete here.
    # paths4 = Paths(base_path, embeddings_path, "CHIME_SVIIvR--Gromet-FN-auto.json", "CHIME_SVIIvR.json", "CHIME_SViiR.json")

    Tester(paths).test()
