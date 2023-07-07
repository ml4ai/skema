package org.ml4ai.skema.text_reading.apps

import org.clulab.utils.{FileUtils, Logging}
import org.ml4ai.skema.text_reading.CosmosTextReadingPipeline
import org.ml4ai.skema.text_reading.utils.{ArgsConfig, CommandLineArgumentParser}

import scala.util.Using
import scopt.OParser

import java.io.{File, FileOutputStream, PrintWriter}


object AnnotateCosmosJsonFiles extends App with Logging{

  val parser = CommandLineArgumentParser.buildParser(getClass.getSimpleName.dropRight(1))

  OParser.parse(parser, args, ArgsConfig()) match {
    case Some(config) =>
      val textReadingPipeline = new CosmosTextReadingPipeline(contextWindowSize = 3) // TODO Add the window parameter to the configuration file

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
            Using(FileUtils.printWriterFromFile(outputFile)) { printWriter =>
              printWriter.println(jsonContents)
              logger.info(s"Wrote output to ${outputFile.getAbsolutePath}")
            }
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
