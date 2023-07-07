package org.ml4ai.skema.text_reading

import org.clulab.odin.Mention
import org.ml4ai.skema.text_reading.scenario_context.{ContextEngine, SentenceIndexOrderer}
import org.ml4ai.skema.text_reading.serializer.SkemaJSONSerializer

/**
  * Runs extraction, grounding and context on input string objects
  */
class TextReadingPipelineWithContext() extends TextReadingPipeline {

  /**
    * Extracts mentions and runs the context engine over the extractions
    * @param inputText text to annotate
    * @param contextWindowSize Size of the context window
    * @return extractions with scenario context
    */
  def extractMentionsWithContext(inputText: String, contextWindowSize: Int): Seq[Mention] = {
    val mentions = extractMentions(inputText, None)._2
    val scenarioContextEngine = new ContextEngine(contextWindowSize, mentions, SentenceIndexOrderer)
    val mentionsWithScenarioContext = mentions.map(scenarioContextEngine.resolveContext)

    mentionsWithScenarioContext
  }

  /**
    * Extracts the mentions and serializes them into a json string
    *
    * @param inputText Text to annotate
    * @param contextWindowSize Size of the context window
    * @return string with the json representation of the extractions and the document annotations
    */
  def extractMentionsWithContextAndSerialize(inputText: String, contextWindowSize: Int): String = {
    val mentions = extractMentionsWithContext(inputText, contextWindowSize)
    val json = ujson.write(SkemaJSONSerializer.serializeMentions(mentions))

    json
  }
}
