package org.ml4ai.skema.text_reading.scenario_context

import org.clulab.odin.{EventMention, Mention, RelationMention, TextBoundMention}
import org.ml4ai.skema.text_reading.attachments.{LocationContextAttachment, TimeContextAttachment}

class ContextEngine(windowSize:Int, documentMentions:Iterable[Mention], orderer: MentionOrderer) {


  private val contextLabels = Set("Location", "Date")


 // Select TB mentions of Location and Date type
  private val candidateMentions = documentMentions.filter{
     case tb:TextBoundMention if contextLabels contains tb.label => true
     case _ => false
  }

  private val mentionsMap: Map[Int, Iterable[Mention]] = candidateMentions.groupBy(orderer.resolveLinearOrder)

  private val maxIndex = mentionsMap.keys.max


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
        }.groupBy(_.label).map {
          case (label, mentions) if label == "Location" =>
            new LocationContextAttachment(mentions)
          case (label, mentions) if label == "Date" => new TimeContextAttachment(mentions)
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
