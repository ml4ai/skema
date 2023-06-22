package org.ml4ai.skema.text_reading.scenario_context

import org.clulab.odin.Mention

/**
  * Trait used to implement a partial order to mentions contained in a "document"
  * Document could mean a plan string with multiple sentences, a pdf, a cosmos json, etc.
  */
trait MentionOrderer {
  /**
    * Takes a mention and linearizes its order for context assignment.
    * This function will encapsulate the ordering logic in the case of a non-trivial ordering, i.e. reading from a Cosmos Json file
    * @param m mention to resolve its linear order
    * @return Linear order of appearance in the encapsulating document. Roughly equivalent of the sentence number in which the mention appears if the source document is a plain text file
    */
  def resolveLinearOrder(m:Mention):Int
}

/**
  * Default implementation that just looks at the sentence index to resolve the ordering.
  * This implementation could be used when the input document is a plain string, without any additional structure
  */
object SentenceIndexOrderer extends MentionOrderer{
  /**
    * Takes a mention and linearizes its order for context assignment.
    * This function will encapsulate the ordering logic in the case of a non-trivial ordering, i.e. reading from a Cosmos Json file
    *
    * @param m mention to resolve its linear order
    * @return Sentence index of the parameter
    */
  override def resolveLinearOrder(m: Mention): Int = m.sentence
}

