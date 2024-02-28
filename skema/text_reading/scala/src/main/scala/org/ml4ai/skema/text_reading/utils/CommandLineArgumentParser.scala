package org.ml4ai.skema.text_reading.utils

import scopt.OParser

import java.io.File

/**
  * Command line arguments parsed by the text
  */
case class ArgsConfig(
                       outDir: File = new File("."),
                       annotateGrounding: Boolean = false,
                       contextWindowSize: Int = 3,
                       contextEngineType: String = "heuristic",
                       inputFiles: Seq[File] = Seq(),
                     )

/**
  * Fabric of command line arguments parsers for our app classes
  */
object CommandLineArgumentParser {
  /**
    * Builds a parser for entry-point apps with customizable program name
    * @param appName Name of the app
    * @return configured parser
    */
  def buildParser(appName:String): OParser[Unit, ArgsConfig] = {
    val builder = OParser.builder[ArgsConfig]

    val parser = {
      import builder._
      OParser.sequence(
        programName(appName),
        head(appName, "1.0"),
        // option -f, --foo
        opt[File]('o', "output_dir")
          .required()
          .action((x, c) => c.copy(outDir = x)).withFallback(() => new File("."))
          .text("directory to write the output files to"),
        opt[Int]('w', "contextWindowSize")
          .optional()
          .action((w, args) => args.copy(contextWindowSize = w)).withFallback(() => 3),
        opt[String]('w', "contextEngineType")
          .optional()
          .action((t, args) => args.copy(contextEngineType = t)).withFallback(() => "heuristic"),
        opt[Unit]('a', "annotateGrounding")
          .action((_, c) => c.copy(annotateGrounding = true))
          .text("Writes grounding file for human annotation"),
        // more options here...
        arg[Seq[File]]("<input file 1>...")
          .unbounded()
          .action((x, c) => c.copy(inputFiles = x))
          .text("files to annotate"),
      )
    }

    parser
  }
}
