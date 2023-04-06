package org.ml4ai.skema.text_reading.scenario_context

import edu.stanford.nlp.ie.machinereading.structure.EventMention
import org.apache.poi.openxml4j.exceptions.InvalidOperationException
import org.clulab.odin.Mention
import org.ml4ai.skema.text_reading.attachments.EventMentionAttachment

/**
  * Implementation to resolve ordering for mentions that come from a Cosmos Json file.
  * Will use the `MentionLocationAttachment` to inform the final linear order using page numbers
  *
  * @param ms All the mentions from the Cosmos Json, used to establish the order bounds
  */
class LinearOrderer(ms:Iterable[Mention]) extends MentionOrderer{

  private def getMentionCosmosLocations(m:Mention) = m.attachments.filter(_.isInstanceOf[EventMentionAttachment]).head.asInstanceOf[EventMentionAttachment]
  private case class OrderingHelper(page:Int, block:Int, sent:Int, sid:Int, s:Int  = 0)

  // Will use the `MentionLocationAttachment` on each mention to generate an offset for each (page, block)
  private val offsets = {
    ms.toSeq.map(
      m => {
        if (!m.attachments.exists(_.isInstanceOf[Mention]) && !m.document.isInstanceOf[EventMention])
          throw new InvalidOperationException("This is not an Event Mention or is without MentionLocationAttachment")

        val location = getMentionCosmosLocations(m)
        val page = location.pageNum.head
        val block = location.blockIdx.head
        val numSentences = m.document.sentences.length
        val sentenceId = m.sentence
        val id = m.document.id
        val text = m.document.text
        OrderingHelper(page, block, numSentences, sentenceId)
      }
    ).distinct.sortBy{
      case OrderingHelper(p,b,s,sid, _) => (p, b, s, sid)
    }.scan(OrderingHelper(0, 0, 0, 0))(
      (prev, curr) => {
        OrderingHelper(curr.page, curr.block, prev.s, curr.sid, prev.s+curr.sent)
      }
    ).map{case OrderingHelper(p, b, s, sid,_) => (p, b, sid) -> s}.toMap

  }

  /**
    * Takes a mention and linearizes its order for context assignment.
    * This function will encapsulate the ordering logic in the case of a non-trivial ordering, i.e. reading from a Cosmos Json file
    *
    * @param m mention to resolve its linear order
    * @return Linear order of appearance in the encapsulating document. Roughly equivalent of the sentence number in which the mention appears if the source document is a plain text file
    */
  override def resolveLinearOrder(m: Mention): Int = {
    val thisMention = getMentionCosmosLocations(m)
    val offset = offsets((thisMention.pageNum.head, thisMention.blockIdx.head)) // TODO: Make these scalars, not arrays
    offset + m.sentence
  }
}

