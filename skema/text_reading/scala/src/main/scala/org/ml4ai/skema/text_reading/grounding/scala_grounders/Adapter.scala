package org.ml4ai.skema.grounding.scala_grounders

import org.clulab.scala_grounders.grounding.GroundingConfig
import org.ml4ai.skema.text_reading.grounding.Grounder
import org.ml4ai.skema.text_reading.grounding.GroundingCandidate
import org.ml4ai.skema.text_reading.grounding.GroundingConcept
import com.typesafe.config.Config
import org.clulab.scala_grounders.grounding.SequentialGrounder
import org.clulab.scala_grounders.model.DKG
import org.clulab.scala_grounders.model.DKGSynonym
import com.typesafe.config.ConfigFactory
import org.clulab.scala_grounders.using
import org.clulab.scala_grounders.model.DKG
import scala.io.Source


/**
  * This class adapts the data definitions from this project to work with scala-grounder's definition
  * Concretely, the changes needed are:
  *   - SKEMA's GroundingConcept to scala-grounder's DKG (avalaible in `fromConceptToDKG`)
  *   - scala-grounder's DKG to SKEMA's GroundingConcept (avalaible in `fromDKGToConcept`)
  *   - Create the scala-grounder Grounder (`grounder = SequentialGrounder()`)
  *   - Changing `groundingCandidates` to call the right method from the scala-grounder side
  *
  * @param groundingConcepts -> The concepts which we will use to do the grounding
  *                             Every candidate text for grounding (i.e. any text that we 
  *                             want to ground) will be grounded on these concepts 
  *                             (Note: depending on the implementation, it is possible that
  *                             none of these groundingConcepts candidates are suitable, so 
  *                             we might not return anything; however, we will never return
  *                             a concept that is outside this)
  */
class ScalaGroundersAdapter(groundingConcepts: Seq[GroundingConcept]) extends Grounder {
  lazy val concepts = groundingConcepts.map(fromConceptToDKG)
  lazy val grounder = SequentialGrounder().mkFast(concepts)
  def groundingCandidates(texts: Seq[String], k: Int): Seq[Seq[GroundingCandidate]] = {
    texts.map { text =>
      // TODO Maybe provide additional context (useful for NeuralGrounder)
      grounder.ground(text, None, concepts, k)
        .map { result => 
          GroundingCandidate(fromDKGToConcept(result.dkg), result.score, details = Some(result.groundingDetails.grounderName)) 
        }
        .force.toSeq
    }
  }

  /**
    * Transform a SKEMA's `GroundingConcept` to a scala-grounders' `DKG`
    * They have similar meanings, so the map is 1:1
    *
    * @param concept
    * @return
    */
  def fromConceptToDKG(concept: GroundingConcept): DKG = {
    DKG(concept.id, concept.name, concept.description, concept.synonyms.map { synonyms => synonyms.map { s => DKGSynonym(s, None) } }.getOrElse(Seq.empty))
  }

  /**
    * Transform a scala-grounder' `DKG` to SKEMA's `GroundingConcept`
    * They have similar meanings, so the map is 1:1
    *
    * @param dkg
    * @return
    */
  def fromDKGToConcept(dkg: DKG): GroundingConcept = {
    GroundingConcept(dkg.id, dkg.name, dkg.description, Option(dkg.synonyms.map(_.value)), None)
  }

}
/**
  * Provide altenatives way of creating a `ScalaGroundersAdapter`
  */
object ScalaGroundersAdapter {
  def apply(groundingConcepts: Seq[GroundingConcept]): ScalaGroundersAdapter = new ScalaGroundersAdapter(groundingConcepts)
  def fromDkgs(dkgs: Seq[DKG]): ScalaGroundersAdapter = new ScalaGroundersAdapter(dkgs.map(dkg => GroundingConcept(dkg.id, dkg.name, dkg.description, Option(dkg.synonyms.map(_.value)), None)))
  def fromFile(groundingConceptsPath: String): ScalaGroundersAdapter = {
    val concepts = using(Source.fromFile(groundingConceptsPath)) { it =>
      ujson.read(it.mkString).arr.map(it => DKG.fromJson(it))
    }
    ScalaGroundersAdapter.fromDkgs(concepts)
  }
}
