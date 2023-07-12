package org.ml4ai.skema.test

import org.clulab.utils.FileUtils
import org.ml4ai.skema.text_reading.CosmosTextReadingPipeline

import java.io.File

class ExtractionFromCosmosTest extends Test {

  behavior of "cosmos extraction"

  // This is simulating what the HomeController does.
  it should "get the same answer as before" in {
    println(new File(".").getAbsolutePath)
    val jsonPaths = Array(
      "./src/test/resources/documents_5Febbuckymodel_webdocs--COSMOS-data.json",
      "./src/test/resources/documents_5FebCHIME_SIR--COSMOS-data.json",
      "./src/test/resources/documents_5FebCHIME_SVIIvR--COSMOS-data.json"
    )
    val cosmosPipeline = new CosmosTextReadingPipeline(contextWindowSize = 3)
    val expected = {
      val ujsonValues = jsonPaths.map { jsonPath =>
        val json = cosmosPipeline.extractMentionsFromJsonAndSerialize(jsonPath)
        val ujsonValue = ujson.read(json)

        ujsonValue
      }
      val ujsonArray = ujson.Arr.from(ujsonValues)
      val result = ujson.write(ujsonArray)

      result
    }
    val actual = {
      val ujsonValues = jsonPaths.map { jsonPath =>
        val json = FileUtils.getTextFromFile(jsonPath)
        val inputUjsonValue = ujson.read(json)
        val outputUjsonValue = cosmosPipeline.extractMentionsFromJsonAndSerialize(inputUjsonValue)

        outputUjsonValue
      }
      val ujsonArray = ujson.Arr.from(ujsonValues)
      val result = ujson.write(ujsonArray)

      result
    }

    actual should be (expected)
  }
}
