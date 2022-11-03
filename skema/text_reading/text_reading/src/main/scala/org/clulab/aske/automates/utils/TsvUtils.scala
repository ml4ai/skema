package org.clulab.aske.automates.utils

import org.clulab.utils.Sourcer

object TsvUtils {

  def loadFromOneColumnTSV(resourcePath: String): Array[String] = {
    val bufferedSource = Sourcer.sourceFromResource(resourcePath)
    val freqWordsIter = for (
      line <- bufferedSource.getLines
    ) yield line.trim

    val freqWords = freqWordsIter.toArray
    bufferedSource.close()
    freqWords
  }
}
