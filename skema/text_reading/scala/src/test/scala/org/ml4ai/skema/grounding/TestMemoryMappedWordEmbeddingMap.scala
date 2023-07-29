package org.ml4ai.skema.grounding

import org.clulab.embeddings.CompactWordEmbeddingMap
import org.clulab.utils.InputStreamer
import org.ml4ai.skema.test.Test
import org.ml4ai.skema.text_reading.grounding.MemoryMappedWordEmbeddingMap

class TestMemoryMappedWordEmbeddingMap extends Test {

  behavior of "MemoryMappedWordEmbeddingMap"

  it should "produce the same results as the im-memory variety" in {
    val resourcePath = "/org/clulab/epimodel/epidemiology_embeddings_model.ser" // fits into 2 buffers
//    val resourcePath = "/org/clulab/spaceweather/spaceweather_model_unigram.ser" // fits into one buffer
    val inputStreamer = new InputStreamer(this)
    val inputStream = inputStreamer.getResourceAsStream(resourcePath)
    val buildType: CompactWordEmbeddingMap.BuildType = CompactWordEmbeddingMap.loadSer(inputStream)

    val compactWordEmbeddingsMap = new CompactWordEmbeddingMap(buildType)
    val memoryMappedWordEmbeddingsMap = new MemoryMappedWordEmbeddingMap(buildType)
    val knownKeys = compactWordEmbeddingsMap.knownKeys

    knownKeys.foreach { word =>
      val compactArray = compactWordEmbeddingsMap.get(word).get
      val memoryMappedArray = memoryMappedWordEmbeddingsMap.get(word).get

      compactArray should equal (memoryMappedArray)
    }
  }
}
