package org.ml4ai.skema.text_reading.apps

import org.clulab.utils.Logging
import org.ml4ai.skema.text_reading.CosmosTextReadingPipeline
import scopt.OParser

import java.io.{File, FileOutputStream, PrintWriter}
case class ArgsConfig(
                   outDir: File = new File("."),
                   inputFiles: Seq[File] = Seq(),
)

object AnnotateCosmosJsonFiles extends App with Logging{

  val builder = OParser.builder[ArgsConfig]

  val parser = {
    import builder._
    OParser.sequence(
      programName("AnnotateCosmosJsonFiles"),
      head("AnnotateCosmosJsonFiles", "1.0"),
      // option -f, --foo
      opt[File]('o', "output_dir")
        .required()
        .action((x, c) => c.copy(outDir = x)).withFallback( () => new File("."))
        .text("directory to write the output files to"),
      // more options here...
      arg[Seq[File]]("<input file 1>...")
        .unbounded()
        .action((x, c) => c.copy(inputFiles = x))
        .text("files to annotate"),
    )
  }

  OParser.parse(parser, args, ArgsConfig()) match {
    case Some(config) =>
      val textReadingPipeline = new CosmosTextReadingPipeline

      logger.info(s"Starting process with ${config.inputFiles} arguments")

      // If an argument is a directory, look at all the inner files, otherwise consider it a file to annotate
      val pathsToAnnotate = config.inputFiles flatMap {
        p => {
          if(p.isDirectory)
            p.listFiles()
          else
            Seq(p)
        }

      }

      for{
        inputFile <- pathsToAnnotate.par
        if inputFile.getName.endsWith(".json")
      } {

        try {
          if(inputFile.exists()){
            val outputFile = new File(config.outDir, "extractions_" + inputFile.getName)
            logger.info(s"Extraction mentions from ${inputFile.getAbsolutePath}")
            val jsonContents = textReadingPipeline.extractMentionsFromJsonAndSerialize(inputFile.getAbsolutePath)
            val writer = new PrintWriter(new FileOutputStream(outputFile))
            writer.write(jsonContents)
            writer.close()
            logger.info(s"Wrote output to ${outputFile.getAbsolutePath}")
          }
          else
            logger.error(s"Didn't find ${inputFile.getAbsolutePath}")
        } catch {
          case e:Exception =>
            logger.error(s"Failed annotating: ${inputFile.getAbsolutePath} with $e")
        }

      }

      logger.info("Finished")
    case _ =>
      // arguments are bad, error message will have been displayed
  }





}
