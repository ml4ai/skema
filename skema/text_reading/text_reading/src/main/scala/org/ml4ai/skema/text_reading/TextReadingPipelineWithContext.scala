package org.ml4ai.skema.text_reading

import org.clulab.odin.Mention
import org.ml4ai.skema.text_reading.scenario_context.{ContextEngine, SentenceIndexOrderer}
import org.ml4ai.skema.text_reading.serializer.SkemaJSONSerializer

/**
  * Runs extraction, grounding and context on input string objects
  */
class TextReadingPipelineWithContext(contextWindowSize:Int) extends TextReadingPipeline{

  /**
    * Extracts mentions and runs the context engine over the extractions
    * @param inputText text to annotate
    * @return extractions with scenario context
    */
  def extractMentionsWithContext(inputText:String):Seq[Mention] ={
    val mentions = this.extractMentions(inputText, None)._2

    // Resolve scenario context
    val scenarioContextEngine = new ContextEngine(windowSize = contextWindowSize, mentions, SentenceIndexOrderer)
    val mentionsWithScenarioContext = mentions map scenarioContextEngine.resolveContext
    mentionsWithScenarioContext
  }

  /**
    * Extracts the mentions and serializes them into a json string
    *
    * @param inputText Text to annotate
    * @return string with the json representation of the extractions and the document annotations
    */
  def extractMentionsWithContextAndSerialize(inputText: String): String = ujson.write(SkemaJSONSerializer.serializeMentions(this.extractMentionsWithContext(inputText)))

}
