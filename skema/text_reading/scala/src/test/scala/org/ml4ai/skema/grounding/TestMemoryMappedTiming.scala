package org.ml4ai.skema.grounding

import org.clulab.embeddings.CompactWordEmbeddingMap
import org.clulab.utils.{InputStreamer, Timer}
import org.ml4ai.skema.test.Test
import org.ml4ai.skema.text_reading.grounding.MemoryMappedWordEmbeddingMap

class TestMemoryMappedTiming extends Test {

  behavior of "MemoryMappedWordEmbeddingMap"

  it should "be slower than the in-memory variety" in {
    val resourcePath = "/org/clulab/epimodel/epidemiology_embeddings_model.ser" // fits into 2 buffers
//    val resourcePath = "/org/clulab/spaceweather/spaceweather_model_unigram.ser" // fits into one buffer
    val inputStreamer = new InputStreamer(this)
    val inputStream = inputStreamer.getResourceAsStream(resourcePath)
    val buildType: CompactWordEmbeddingMap.BuildType = CompactWordEmbeddingMap.loadSer(inputStream)

    val compactWordEmbeddingsMap = new CompactWordEmbeddingMap(buildType)
    val memoryMappedWordEmbeddingsMap = new MemoryMappedWordEmbeddingMap(buildType)
    val knownKeys = compactWordEmbeddingsMap.knownKeys

    val compactTimer = new Timer("compactWordEmbeddingsMap")
    compactTimer.time {
      knownKeys.foreach { word =>
        compactWordEmbeddingsMap.get(word).get
      }
    }
    println(compactTimer)

    val memoryMappedTimer = new Timer("memoryMappedWordEmbeddingsMap")
    memoryMappedTimer.time {
      knownKeys.foreach { word =>
        memoryMappedWordEmbeddingsMap.get(word).get
      }
    }
    println(memoryMappedTimer)

    memoryMappedTimer.elapsedTime should be > (compactTimer.elapsedTime)
  }
}
