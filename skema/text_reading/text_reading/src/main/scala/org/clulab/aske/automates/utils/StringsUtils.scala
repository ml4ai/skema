package org.clulab.aske.automates.utils

import scala.io.Source

object StringsUtils {
  def loadStrings(path: String): Array[String] = {
    val source = Source.fromFile(path)
    val lines = source.getLines().toArray
    source.close()
    lines
  }

  def loadStringsFromResource(resourcePath: String): Array[String] = {
    org.clulab.utils.FileUtils.getTextFromResource(resourcePath).split("\n")
  }
}
