package org.ml4ai.skema.text_reading

import org.clulab.odin.Mention
import org.clulab.utils.FileUtils
import org.ml4ai.skema.text_reading.scenario_context.{ContextEngine, CosmosOrderer, SentenceIndexOrderer}
import org.ml4ai.skema.text_reading.serializer.SkemaJSONSerializer

import java.io.File
import scala.io.Source
import scala.util.{Failure, Success, Using}

class PlainTextFileTextReadingPipeline(contextWindowSize:Int) extends TextReadingPipeline {

  /**
    * Runs the textReadingPipeline over a plain text file
    *
    * @param filePath Path to the text file to annotate
    * @return Mentions extracted by the TR textReadingPipeline
    */
  def extractMentionsFromTextFile(filePath:String):Seq[Mention] = {
    val text = FileUtils.getTextFromFile(filePath)

    logger.info(s"Starting annotation of $filePath")
    val fileName = new File(filePath).getName
    val mentions = this.extractMentions(text, Some(fileName))._2  // TODO Make this a case class for legibility

    // Resolve scenario context
    val scenarioContextEngine = new ContextEngine(windowSize = contextWindowSize, mentions, SentenceIndexOrderer)
    val mentionsWithScenarioContext = mentions map scenarioContextEngine.resolveContext
    logger.info(s"Finished annotation of $filePath")
    mentionsWithScenarioContext
  }

  /**
    * Extracts the mentions and serializes them into a json string
    *
    * @param filePath Path to the plain text file to annotate
    * @return string with the json representation of the extractions and the document annotations
    */
  def extractMentionsFromTextFileAndSerialize(filePath:String): String = ujson.write(SkemaJSONSerializer.serializeMentions(this.extractMentionsFromTextFile(filePath)))

}
