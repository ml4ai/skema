package org.ml4ai.skema.grounding

import org.ml4ai.skema.test.Test

import java.nio.{Buffer, ByteBuffer, ByteOrder}

class TestMemoryMappedWordEmbeddingMap extends Test {

  behavior of "float buffer"

  it should "buffer" in {
    val byteBuffer = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder())
    val floatBuffer = byteBuffer.asFloatBuffer
    val array = new Array[Float](2)

    1.to(10).foreach { index =>
      array(0) = index.toFloat
      array(1) = (index + 1).toFloat
      floatBuffer.put(array)
      (floatBuffer: Buffer).flip()

      val float1 = byteBuffer.getFloat
      float1 should be (array(0))

      val float2 = byteBuffer.getFloat
      float2 should be (array(1))

      (floatBuffer: Buffer).clear()
      (byteBuffer: Buffer).clear()
    }
  }
}
