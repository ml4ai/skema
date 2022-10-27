package org.clulab.aske.automates.utils

import scala.io.Source

object TsvUtils {

  def loadFromOneColumnTSV(resourcePath: String): Array[String] = {
    val bufferedSource = Source.fromFile(resourcePath)
    val freqWordsIter = for (
      line <- bufferedSource.getLines
    ) yield line.trim

    val freqWords = freqWordsIter.toArray
    bufferedSource.close()
    freqWords
  }
}
