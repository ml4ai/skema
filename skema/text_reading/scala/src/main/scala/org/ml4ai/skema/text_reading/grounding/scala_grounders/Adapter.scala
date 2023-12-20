import org.clulab.scala_grounders.grounding.GroundingConfig
import org.ml4ai.skema.text_reading.grounding.Grounder
import org.ml4ai.skema.text_reading.grounding.GroundingCandidate
import org.ml4ai.skema.text_reading.grounding.GroundingConcept
import com.typesafe.config.Config
import org.clulab.scala_grounders.grounding.SequentialGrounder
import org.clulab.scala_grounders.model.DKG
import org.clulab.scala_grounders.model.DKGSynonym
import com.typesafe.config.ConfigFactory

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
  lazy val grounder = SequentialGrounder()
  def groundingCandidates(texts: Seq[String], k: Int): Seq[Seq[GroundingCandidate]] = {
    val concepts = groundingConcepts.map(fromConceptToDKG)
    texts.map { text =>
      grounder.ground(text, concepts, k)
        .map { result => 
          println(result)
          GroundingCandidate(fromDKGToConcept(result.dkg), result.score) 
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
  private def fromConceptToDKG(concept: GroundingConcept): DKG = {
    DKG(concept.id, concept.name, concept.description, concept.synonyms.map { synonyms => synonyms.map { s => DKGSynonym(s, None) } }.getOrElse(Seq.empty))
  }

  /**
    * Transform a scala-grounder' `DKG` to SKEMA's `GroundingConcept`
    * They have similar meanings, so the map is 1:1
    *
    * @param dkg
    * @return
    */
  private def fromDKGToConcept(dkg: DKG): GroundingConcept = {
    GroundingConcept(dkg.id, dkg.name, dkg.description, Option(dkg.synonyms.map(_.value)), None)
  }

}

object AdHocExample extends App {
  val gcs = Seq(
    GroundingConcept(
      id          = "id1",
      name        = "dog",
      description = Some("this is a cute dog"),
      synonyms    = None,
      embedding   = None
    ), 
    GroundingConcept(
      id          = "id2",
      name        = "cat",
      description = Some("this is a cute cat"),
      synonyms    = None,
      embedding   = None
    ), 
    GroundingConcept(
      id          = "id3",
      name        = "dog cat",
      description = Some("here we have a dog and a cat"),
      synonyms    = None,
      embedding   = None
    ), 
    GroundingConcept(
      id          = "id4",
      name        = "cat",
      description = Some("this is a cute cat"),
      synonyms    = None,
      embedding   = None
    ), 
  )

  val sga = new ScalaGroundersAdapter(gcs)
  // println(sga.grounder.components.toList.toSeq.map(_.getName))
  // val result = sga.groundingCandidates(Seq("dog dog dog"), 10)
  val result = sga.groundingCandidates(Seq("dog"), 10)
  // result.head.foreach(println)
  // println(ConfigFactory.load())
} 
