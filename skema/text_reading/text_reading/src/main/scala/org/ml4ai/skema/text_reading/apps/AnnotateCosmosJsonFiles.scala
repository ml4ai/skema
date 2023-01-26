package org.ml4ai.skema.text_reading.apps

import org.clulab.utils.Logging
import org.ml4ai.skema.text_reading.CosmosTextReadingPipeline

import java.io.{File, FileOutputStream, PrintWriter}

object AnnotateCosmosJsonFiles extends App with Logging{

    private lazy val textReadingPipeline = new CosmosTextReadingPipeline

    logger.info(s"Starting process with ${args.length} arguments")

    for{
      path <- args.par
      if path.endsWith(".json")
    } {
      val inputFile = new File(path)
      if(inputFile.exists()){
        val outputFile = new File("extractions_" + inputFile.getName)
        logger.info(s"Extraction mentions from ${inputFile.getAbsolutePath}")
        val jsonContents = textReadingPipeline.serializeToJson(path)
        val writer = new PrintWriter(new FileOutputStream(outputFile))
        writer.write(jsonContents)
        writer.close()
        logger.info(s"Wrote output to ${outputFile.getAbsolutePath}")
      }
      else
        logger.error(s"Didn't find ${inputFile.getAbsolutePath}")

    }

    logger.info("Finished")


}
