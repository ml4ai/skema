package org.ml4ai.skema.text_reading.grounding

import org.clulab.embeddings.{CompactWordEmbeddingMap, WordEmbeddingMap}

import java.io.{ByteArrayOutputStream, DataOutputStream, File, FileOutputStream, OutputStream, RandomAccessFile}
import java.nio.{ByteBuffer, FloatBuffer}
import java.nio.channels.FileChannel
import scala.collection.mutable.{HashMap => MutableHashMap}
import scala.util.Using

// protected type ImplMapType = MutableHashMap[String, Int]
// case class BuildType(map: ImplMapType, array: Array[Float], unknownArray: Option[Array[Float]], columns: Int)
class MemoryMappedWordEmbeddingMap(buildType: CompactWordEmbeddingMap.BuildType) extends WordEmbeddingMap {
  protected val map: MutableHashMap[String, Int] = buildType.map // (word -> row)
  val columns: Int = buildType.columns
  val unkEmbeddingOpt: Option[IndexedSeq[Float]] = buildType.unknownArray.map { inside =>
    inside: IndexedSeq[Float]
  }
  val floatBuffer = {
    val file = {
      val file = File.createTempFile("skema", ".tmp")

      file.deleteOnExit()
      file
    }

    // https://stackoverflow.com/questions/9346746/convert-float-to-byte-to-float-again
    // https://stackoverflow.com/questions/12132595/mappedbytebuffer-asfloatbuffer-vs-in-memory-float-performance
    val array = buildType.array
    val floatBuffer = FloatBuffer.wrap(array)
    // Have a floatBuffer and just transfer a few at a time.
    val byteBuffer = ByteBuffer.allocate(array.length * 4)
//    byteBuffer.asFloatBuffer.put(array)
    byteBuffer.asFloatBuffer.put(floatBuffer)
    // Something about native order

    Using.resource(new FileOutputStream(file)) { fileOutputStream =>
      // Could just do it a row at a time instead.
      fileOutputStream.write(byteBuffer.array)
    }

    val randomAccessFile = new RandomAccessFile(file, "r")
    val channel = randomAccessFile.getChannel
    val size = channel.size()
    val mappedMemoryBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
    val floatBuffer2 = mappedMemoryBuffer.asFloatBuffer

    floatBuffer2
  }

  /** The dimension of an embedding vector */
  override val dim: Int = columns

  // Be careful because this word may not be sanitized!
  override def isOutOfVocabulary(word: String): Boolean = !map.contains(word)

  protected def get(row: Int): IndexedSeq[Float] = {
    val array = new Array[Float](columns)

    floatBuffer.get(row * columns, array)
    array
  }

  override def get(word: String): Option[IndexedSeq[Float]] = {
    map.get(word).map { row =>
      get(row)
    }
  }

  override def getOrElseUnknown(word: String): IndexedSeq[Float] = {
    get(word).getOrElse(
      unkEmbeddingOpt.getOrElse(
        throw new RuntimeException("ERROR: can't find embedding for the unknown token!")
      )
    )
  }

  override def keys: Set[String] = map.keys.toSet

  override def unknownEmbedding: IndexedSeq[Float] = unkEmbeddingOpt.get

  protected def add(dest: Array[Float], srcRow: Int): Unit = {
    val srcArray = get(srcRow)
    var i = 0 // optimization

    while (i < columns) {
      dest(i) += srcArray(i)
      i += 1
    }
  }

  override def makeCompositeVector(text: Iterable[String]): Array[Float] = {
    val total = new Array[Float](columns) // automatically initialized to zero

    text.foreach { word =>
      // This therefore skips the unknown words, which may not be the right strategy.
      map.get(word).foreach { index => add(total, index) }
    }
    WordEmbeddingMap.norm(total)
    total
  }

  protected def addWeighted(dest: Array[Float], srcRow: Int, weight: Float): Unit = {
    val srcArray = get(srcRow)
    var i = 0 // optimization

    while (i < columns) {
      dest(i) += srcArray(i) * weight
      i += 1
    }
  }

  override def makeCompositeVectorWeighted(text: Iterable[String], weights: Iterable[Float]): Array[Float] = {
    val total = new Array[Float](columns) // automatically initialized to zero

    text.zip(weights).foreach { case (word, weight) =>
      // This therefore skips the unknown words, which may not be the right strategy.
      map.get(word).foreach { index => addWeighted(total, index, weight) }
    }
    WordEmbeddingMap.norm(total)
    total
  }

  def dotProduct(row1: Int, row2: Int): Float = {
    val array1 = get(row1)
    val array2 = get(row2)
    var sum = 0f
    var i = 0 // optimization

    while (i < columns) {
      sum += array1(i) * array2(i)
      i += 1
    }
    sum
  }

  override def avgSimilarity(texts1: Iterable[String], texts2: Iterable[String]): Float = {
    var sum = 0f // optimization
    var count = 0 // optimization

    texts1.foreach { text1 =>
      val row1Opt = map.get(text1)

      if (row1Opt.isDefined) {
        texts2.foreach { text2 =>
          val row2Opt = map.get(text2)

          if (row2Opt.isDefined) {
            sum += dotProduct(row1Opt.get, row2Opt.get)
            count += 1
          }
        }
      }
    }
    if (count != 0) sum / count
    else 0f
  }

  override def save(filename: String): Unit = ???
}
