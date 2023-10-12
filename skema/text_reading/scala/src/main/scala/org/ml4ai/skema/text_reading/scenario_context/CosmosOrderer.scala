package org.ml4ai.skema.text_reading.scenario_context
import org.clulab.odin.Mention
import org.ml4ai.skema.text_reading.attachments.MentionLocationAttachment

/**
  * Implementation to resolve ordering for mentions that come from a Cosmos Json file.
  * Will use the `MentionLocationAttachment` to inform the final linear order using page numbers
  *
  * @param ms All the mentions from the Cosmos Json, used to establish the order bounds
  */
class CosmosOrderer(ms:Iterable[Mention]) extends MentionOrderer {

  private def getMentionLocation(m:Mention) = m.attachments.filter(_.isInstanceOf[MentionLocationAttachment]).head.asInstanceOf[MentionLocationAttachment]

  private case class OrderingHelper(page:Int, block:Int, sent:Int, s:Int  = 0)

  // Will use the `MentionLocationAttachment` on each mention to generate an offset for each (page, block)
  private val offsets =
    ms.toSeq.map(
      m => {
        if (!m.attachments.exists(_.isInstanceOf[MentionLocationAttachment]))
          throw new IllegalStateException("Mention without MentionLocationAttachment")

        val location = getMentionLocation(m)
        val page = location.pageNum.head
        val block = location.blockIdx.head
        val numSentences = m.document.sentences.length
        OrderingHelper(page, block, numSentences)
      }
    ).distinct.sortBy{
      case OrderingHelper(p,b,s, _) => (p, b, s)
    }.scan(OrderingHelper(0, 0, 0))(
      (prev, curr) => {
        OrderingHelper(curr.page, curr.block, prev.s, prev.s+curr.sent)
      }
    ).map{case OrderingHelper(p, b, s, _) => (p, b) -> s}.toMap

  /**
    * Takes a mention and linearizes its order for context assignment.
    * This function will encapsulate the ordering logic in the case of a non-trivial ordering, i.e. reading from a Cosmos Json file
    *
    * @param m mention to resolve its linear order
    * @return Linear order of appearance in the encapsulating document. Roughly equivalent of the sentence number in which the mention appears if the source document is a plain text file
    */
  override def resolveLinearOrder(m: Mention): Int = {
    val location = getMentionLocation(m)
    val offset = offsets((location.pageNum.head, location.blockIdx.head)) // TODO: Make these scalars, not arrays
    offset + m.sentence
  }
}
