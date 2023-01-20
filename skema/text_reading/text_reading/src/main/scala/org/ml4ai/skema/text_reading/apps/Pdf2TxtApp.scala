package org.ml4ai.skema.text_reading.apps

import org.clulab.pdf2txt.Pdf2txt
import org.clulab.pdf2txt.common.pdf.TextConverter
import org.clulab.pdf2txt.languageModel.GigawordLanguageModel
import org.clulab.pdf2txt.preprocessor.{CasePreprocessor, LigaturePreprocessor, LineBreakPreprocessor, LinePreprocessor, NumberPreprocessor, ParagraphPreprocessor, UnicodePreprocessor, WordBreakByHyphenPreprocessor, WordBreakBySpacePreprocessor}

object Pdf2TxtApp extends App {
  val pdfConverter = new TextConverter()
  val languageModel = GigawordLanguageModel()
  val preprocessors = Array(
    new LinePreprocessor(),
    new ParagraphPreprocessor(),
    new UnicodePreprocessor(),
    new CasePreprocessor(CasePreprocessor.defaultCutoff),
    new NumberPreprocessor(NumberPreprocessor.Hyperparameters()),
    new LigaturePreprocessor(languageModel),
    new LineBreakPreprocessor(languageModel),
    new WordBreakByHyphenPreprocessor(),
    new WordBreakBySpacePreprocessor(languageModel) // This is by default NeverLanguageModel.
  )
  val pdf2txt = new Pdf2txt(pdfConverter, preprocessors)
  val rawTexts = Array(
    "fi gures",
    "o ffi cials",
    "di ffi cult",
    "sta ffi ng",
    "shif ts",
    "speci fi cally",
    "ca not",
    "we ' ve",
    "tra ffi c",
    "fi rst",
    "Here 's",
    "Let 's",
    "hospital s",
    "o ffi cials find sta ffi ng di ffi cult"
  )

  rawTexts.foreach { rawText =>
    val cookedText = pdf2txt.process(rawText, maxLoops = 1)

    println(s"   rawText: $rawText")
    println(s"cookedText: $cookedText")
    println()
  }
}
