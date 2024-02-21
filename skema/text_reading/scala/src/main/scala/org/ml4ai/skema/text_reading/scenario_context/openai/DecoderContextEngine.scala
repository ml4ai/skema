package org.ml4ai.skema.text_reading.scenario_context.openai

import org.clulab.odin.{EventMention, Mention, RelationMention, TextBoundMention}
import org.ml4ai.skema.text_reading.attachments.{LocationContextAttachment, TimeContextAttachment}
import org.ml4ai.skema.text_reading.scenario_context.{ContextEngine, MentionOrderer}

class DecoderContextEngine(windowSize:Int, documentMentions:Iterable[Mention], orderer: MentionOrderer) extends ContextEngine {


  private val contextLabels = Set("Location", "Date")
  private val prompter = new PromptScenarioContext

  override def finalize(): Unit = {
    prompter.closeAll()
  }


  // Select TB mentions of Location and Date type
  private val candidateMentions = documentMentions.filter{
    case tb:TextBoundMention if contextLabels(tb.label) => true
    case _ => false
  }

  private val mentionsMap: Map[Int, Iterable[Mention]] = candidateMentions.groupBy(orderer.resolveLinearOrder)

  private val maxIndex = if(mentionsMap.nonEmpty) mentionsMap.keys.max else 0


  def resolveContext(m:Mention): Mention =  {

    if(m.isInstanceOf[EventMention] || m.isInstanceOf[RelationMention]) {
      val location = orderer.resolveLinearOrder(m)
      val start = Math.max(0, location - windowSize)
      val end = Math.min(maxIndex, location + windowSize)

      val isCandidate =
        (start to end).flatMap {
          sentenceIndex =>
            val contextMentions = mentionsMap.getOrElse(sentenceIndex, Seq.empty)
            contextMentions
        }.groupBy(_.label).exists {
          case ("Location", mentions) if mentions.nonEmpty => true// To optimize, let's restrict ourselves to instances where there is a location or a date in the neighborhood
          case ("Date", mentions) if mentions.nonEmpty => true
          case _ => false
        }

      val contextAttachments =
        if(isCandidate){
          // Get the context of the mention
          val docText = m.document.text.get
          val left = docText.take(m.startOffset) // Operate in offsets because this is the character based index
          val focus = docText.slice(m.startOffset, m.endOffset)
          val right = docText.takeRight(m.endOffset)
          val locationContext = prompter.promptForLocationContext(left, focus, right)
          val temporalContext = prompter.promptForTemporalContext(left, focus, right)
          List(new LocationContextAttachment(locationContext), new TimeContextAttachment(temporalContext))
        }
        else
          Nil

      m match {
        case evt: EventMention =>
          // Add the attachments to the mention
          evt.newWithAttachments(contextAttachments)

        case rel: RelationMention =>
          // Add the attachments to the mention
          rel.newWithAttachments(contextAttachments)
        case el => el // This is just to complete the match statement, should never pass through here
      }
    }
    else
      m

  }

}
