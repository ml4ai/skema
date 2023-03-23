package org.ml4ai.skema.text_reading.grounding

import scala.collection.mutable

class MiraWebApiGrounder(apiEndPoint: String, batchSize: Option[Int] = None) extends Grounder {

  // Cache to store previously resolved items
  // TODO this is a potential memory leak. We need to change for an LRU cache
  private val resolutionCache = mutable.HashMap[String, Seq[GroundingCandidate]]()

  /**
    * Hits the MIRA API with the post request to fetch groundings
    *
    * @param keys to be queried
    * @return groundings for each input key
    */
  private def requestGrounding(keys: Seq[String]): Seq[Seq[GroundingCandidate]] = {
    if (keys.nonEmpty) {
      val batchSize = this.batchSize.getOrElse(keys.size)

      keys.grouped(batchSize).flatMap {
        batchKeys =>
          val requestBody = ujson.Value(batchKeys.map(k => ujson.Obj("text" -> k))).render()

          val r = requests.post(apiEndPoint,
            headers = Map("accepts" -> "application/json", "Content-Type" -> "application/json"),
            data = requestBody)

          ujson.read(r.text).arr.map(
            obj => {
              obj("results").arr.map {
                data =>
                  GroundingCandidate(
                    GroundingConcept(data("curie").str, data("name").str, description = None, synonyms = None, embedding = None),
                    data("score").num.toFloat
                  )
              }
            }
          )
      }.toSeq
    } else
      Seq()
  }

  /**
    * Returns an ordered sequence with the top k grounding candidates for the input
    *
    * @param texts of the extractions to be grounded
    * @param k     number of max candidates to return
    * @return For each input element, ranked list with the top k candidates
    */
  override def groundingCandidates(texts: Seq[String], k: Int): Seq[Seq[GroundingCandidate]] = {
    // Inspect cache of previously resolved elements
    val normalizedKeys = texts map (k => Grounder.normalizeText(k, caseFold = false))
    val missingKeys = normalizedKeys filterNot resolutionCache.contains
    // Build request body with remaining ungrounded concepts
    val groundings = requestGrounding(missingKeys)
    // Update the cache with the missing groundings
    for ((key, gr) <- missingKeys zip groundings) {
      resolutionCache(key) = gr
    }

    // Build return value
    normalizedKeys map resolutionCache
  }
}
