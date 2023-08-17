package org.ml4ai.skema.text_reading.grounding

import org.clulab.embeddings.{CompactWordEmbeddingMap, WordEmbeddingMap}

import java.io.{BufferedOutputStream, File, FileOutputStream, RandomAccessFile}
import java.nio.{Buffer, ByteBuffer, ByteOrder, FloatBuffer}
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
  val (rowsPerBuffer: Int, floatBuffers: IndexedSeq[FloatBuffer]) = mkRowsPerBufferAndFloatBuffers()

  /** The dimension of an embedding vector */
  override val dim: Int = columns

  // Be careful because this word may not be sanitized!
  override def isOutOfVocabulary(word: String): Boolean = !map.contains(word)

  def ceil(numerator: Int, denominator: Int): Int = {
    import scala.math.Integral.Implicits._
    val (quotient, remainder) = numerator /% denominator

    quotient + math.signum(remainder)
  }

  def mkRowsPerBufferAndFloatBuffers(): (Int, IndexedSeq[FloatBuffer]) = {
    val rows: Int = map.size
    val bytesPerRow = columns * java.lang.Float.BYTES
    val rowsPerBuffer = Int.MaxValue / bytesPerRow
    val bufferCount = ceil(rows, rowsPerBuffer)

    // Keep the size of this buffer small so that we don't have two sets of data at once.
    val byteBuffer = ByteBuffer.allocate(bytesPerRow).order(ByteOrder.nativeOrder)
    val floatBuffer = byteBuffer.asFloatBuffer
    val maxOutputBufferSize = 100000
    val outputBufferSize = maxOutputBufferSize - (maxOutputBufferSize % bytesPerRow)

    val floatBuffers = Range(0, bufferCount).map { index =>
      val file = {
        val file = File.createTempFile(s"skema-$index-", ".tmp")

        file.deleteOnExit()
        file
      }

      // https://stackoverflow.com/questions/9346746/convert-float-to-byte-to-float-again
      // https://stackoverflow.com/questions/12132595/mappedbytebuffer-asfloatbuffer-vs-in-memory-float-performance

      {
        val rowRange = Range(index * rowsPerBuffer, math.min((index + 1) * rowsPerBuffer, rows))

        Using.resource(new BufferedOutputStream(new FileOutputStream(file), outputBufferSize)) { outputStream =>
          rowRange.foreach { row =>
            (floatBuffer: Buffer).clear()
            floatBuffer.put(buildType.array, row * columns, columns)
            outputStream.write(byteBuffer.array)
          }
        }
      }

      {
        val randomAccessFile = new RandomAccessFile(file, "r")
        val channel = randomAccessFile.getChannel
        val mappedMemoryBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size).order(ByteOrder.nativeOrder)
        val floatBuffer = mappedMemoryBuffer.asFloatBuffer

        floatBuffer
      }
    }

    (rowsPerBuffer, floatBuffers)
  }

  protected def get(row: Int): IndexedSeq[Float] = {
    val bufferIndex = row / rowsPerBuffer
    val bufferOffset = (row % rowsPerBuffer) * columns
    val floatBuffer = floatBuffers(bufferIndex)
    val array = new Array[Float](columns)

    (floatBuffer: Buffer).position(bufferOffset)
    floatBuffer.get(array)
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
