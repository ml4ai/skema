package org.ml4ai.skema.text_reading.scenario_context

import org.clulab.odin.{EventMention, Mention, RelationMention, TextBoundMention}
import org.ml4ai.skema.text_reading.attachments.{LocationContextAttachment, TimeContextAttachment}

trait ContextEngine{
  def resolveContext(m:Mention): Mention
}

class HeuristicContextEngine(windowSize:Int, documentMentions:Iterable[Mention], orderer: MentionOrderer) extends ContextEngine {


  private val contextLabels = Set("Location", "Date")


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

      val contextAttachments =
        (start to end).flatMap {
          sentenceIndex =>
            val contextMentions = mentionsMap.getOrElse(sentenceIndex, Seq.empty)
            contextMentions
        }.groupBy(_.label).collect {
          case ("Location", mentions) =>
            new LocationContextAttachment(mentions.map(_.text))
          case ("Date", mentions) => new TimeContextAttachment(mentions.map(_.text))
        }.toSeq.distinct

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
