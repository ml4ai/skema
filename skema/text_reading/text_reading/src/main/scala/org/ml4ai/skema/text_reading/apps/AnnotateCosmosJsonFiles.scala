package org.ml4ai.skema.text_reading.apps

import org.clulab.utils.Logging
import org.ml4ai.skema.text_reading.CosmosTextReadingPipeline

import java.io.{File, FileOutputStream, PrintWriter}

object AnnotateCosmosJsonFiles extends App with Logging{

    private val textReadingPipeline = new CosmosTextReadingPipeline

    logger.info(s"Starting process with ${args.length} arguments")

    // If an argument is a directory, look at all the inner files, otherwise consider it a file to annotate
    val pathsToAnnotate = args flatMap {
      path => {
        val p = new File(path)
        
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
          val outputFile = new File("extractions_" + inputFile.getName)
          logger.info(s"Extraction mentions from ${inputFile.getAbsolutePath}")
          val jsonContents = textReadingPipeline.serializeToJson(inputFile.getAbsolutePath)
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


}
