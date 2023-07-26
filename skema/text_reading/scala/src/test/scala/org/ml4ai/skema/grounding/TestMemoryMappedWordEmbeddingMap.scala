package org.ml4ai.skema.grounding

import org.clulab.embeddings.CompactWordEmbeddingMap
import org.clulab.utils.{InputStreamer, Timer}
import org.ml4ai.skema.test.Test
import org.ml4ai.skema.text_reading.grounding.MemoryMappedWordEmbeddingMap

class TestMemoryMappedWordEmbeddingMap extends Test {

  behavior of "MemoryMappedWordEmbeddingMap"

  it should "produce the same results if not memory mapped" in {
//    val resourcePath = "/org/clulab/epimodel/epidemiology_embeddings_model.ser"
    val resourcePath = "/org/clulab/spaceweather/spaceweather_model_unigram.ser"
    val inputStreamer = new InputStreamer(this)
    val inputStream = inputStreamer.getResourceAsStream(resourcePath)
    val buildType: CompactWordEmbeddingMap.BuildType = CompactWordEmbeddingMap.loadSer(inputStream)

    val compactWordEmbeddingsMap = new CompactWordEmbeddingMap(buildType)
    val memoryMappedWordEmbeddingsMap = new MemoryMappedWordEmbeddingMap(buildType)
    val knownKeys = compactWordEmbeddingsMap.knownKeys

    knownKeys.take(100).foreach { word =>
      val compactArray = compactWordEmbeddingsMap.get(word).get
      val memoryMappedArray = memoryMappedWordEmbeddingsMap.get(word).get

      val compactString = compactArray.mkString(", ")
      val memoryMappedString = memoryMappedArray.mkString(", ")

      if (compactString != memoryMappedString)
        println("It isn't matching!")
    }

    val compactTimer = new Timer("compactWordEmbeddingsMap")
    compactTimer.time {
      knownKeys.take(10000).foreach { word =>
        compactWordEmbeddingsMap.get(word).get
      }
    }
    println(compactTimer)

    val memoryMappedTimer = new Timer("memoryMappedWordEmbeddingsMap")
    memoryMappedTimer.time {
      knownKeys.take(10000).foreach { word =>
        memoryMappedWordEmbeddingsMap.get(word).get
      }
    }
    println(memoryMappedTimer)
  }
}
