package org.ml4ai.skema.grounding

import org.ml4ai.skema.test.Test

import java.nio.{Buffer, ByteBuffer}

class TestMemoryMappedWordEmbeddingMap extends Test {

  behavior of "float buffer"

  it should "buffer" in {
    val byteBuffer = ByteBuffer.allocate(8)
    val floatBuffer = byteBuffer.asFloatBuffer
    val array = new Array[Float](1)

    1.to(10).foreach { index =>
      array(0) = index.toFloat
      floatBuffer.put(array, 0, 1)
      (floatBuffer: Buffer).flip()
      val float = byteBuffer.getFloat
      (floatBuffer: Buffer).flip()
//      (floatBuffer: Buffer).reset()

      float should be (index)
    }
  }
}
