package com.petseg.mobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Segmentor(context: Context) {

    private val interpreter: Interpreter

    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    // ARGB overlay colors per class (~67% alpha)
    private val classColors = intArrayOf(
        Color.argb(170, 255, 111,   0),   // 0 = Pet (amber)
        Color.argb(170,   0, 105,  92),   // 1 = Background (teal)
        Color.argb(170, 158, 158, 158),   // 2 = Border (gray)
    )

    companion object {
        const val IMG_SIZE    = 256
        const val NUM_CLASSES = 3
        private const val IN_CHANNELS  = 3
        private const val MODEL_FILE   = "model.tflite"
        private const val THREADS      = 4
    }

    init {
        val opts = Interpreter.Options().apply { numThreads = THREADS }
        interpreter = Interpreter(loadModelFile(context), opts)
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_FILE)
        return FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength,
        )
    }

    /**
     * Scale to 256×256, normalize per-channel (ImageNet stats),
     * write as NHWC float32 ByteBuffer [1, 256, 256, 3].
     */
    fun preprocess(bitmap: Bitmap): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true)
        val buf = ByteBuffer
            .allocateDirect(1 * IMG_SIZE * IMG_SIZE * IN_CHANNELS * Float.SIZE_BYTES)
            .apply { order(ByteOrder.nativeOrder()) }

        val pixels = IntArray(IMG_SIZE * IMG_SIZE)
        scaled.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)

        for (px in pixels) {
            buf.putFloat((Color.red(px)   / 255f - mean[0]) / std[0])
            buf.putFloat((Color.green(px) / 255f - mean[1]) / std[1])
            buf.putFloat((Color.blue(px)  / 255f - mean[2]) / std[2])
        }
        buf.rewind()
        return buf
    }

    /**
     * Run TFLite inference.
     * Input:  [1, 256, 256, 3] float32  (NHWC)
     * Output: [1, 256, 256, 3] float32  (NHWC logits)
     */
    private fun runInference(input: ByteBuffer): Array<Array<Array<FloatArray>>> {
        val out = Array(1) { Array(IMG_SIZE) { Array(IMG_SIZE) { FloatArray(NUM_CLASSES) } } }
        interpreter.run(input, out)
        return out
    }

    /**
     * Argmax over class dim → IntArray[256*256] with class index per pixel.
     */
    private fun decodeMask(logits: Array<Array<Array<FloatArray>>>): IntArray {
        val mask = IntArray(IMG_SIZE * IMG_SIZE)
        for (y in 0 until IMG_SIZE) {
            val row = logits[0][y]
            for (x in 0 until IMG_SIZE) {
                val scores = row[x]
                var best = 0
                for (c in 1 until NUM_CLASSES) if (scores[c] > scores[best]) best = c
                mask[y * IMG_SIZE + x] = best
            }
        }
        return mask
    }

    /** Full pipeline: preprocess → infer → argmax. */
    fun segment(bitmap: Bitmap): IntArray =
        decodeMask(runInference(preprocess(bitmap)))

    /**
     * Alpha-blend colored class mask over a 256×256 copy of [original].
     * Uses setPixels() batch call instead of per-pixel setPixel().
     */
    fun renderOverlay(original: Bitmap, mask: IntArray): Bitmap {
        val base = Bitmap.createScaledBitmap(original, IMG_SIZE, IMG_SIZE, true)
            .copy(Bitmap.Config.ARGB_8888, true)

        // Build overlay pixel array in one pass
        val overlayPixels = IntArray(mask.size) { i ->
            classColors[mask[i].coerceIn(0, NUM_CLASSES - 1)]
        }
        val overlay = Bitmap.createBitmap(IMG_SIZE, IMG_SIZE, Bitmap.Config.ARGB_8888)
        overlay.setPixels(overlayPixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)

        Canvas(base).drawBitmap(overlay, 0f, 0f, Paint().apply {
            alpha = 255   // alpha already encoded in classColors
        })
        return base
    }

    fun close() = interpreter.close()
}
