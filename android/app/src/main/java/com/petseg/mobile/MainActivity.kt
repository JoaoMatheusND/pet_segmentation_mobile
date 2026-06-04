package com.petseg.mobile

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import com.petseg.mobile.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val segmentor by lazy { Segmentor(this) }

    private var currentBitmap: Bitmap? = null
    private var cameraUri: Uri?        = null

    // ── Result launchers (must be registered before onCreate) ─────────────

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { setSourceImage(it) } }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { ok: Boolean -> if (ok) cameraUri?.let { setSourceImage(it) } }

    private val cameraPermLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted: Boolean ->
        if (granted) launchCamera() else showSnack("Camera permission denied")
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        binding.btnCamera.setOnClickListener  { requestCameraOrLaunch() }
        binding.btnGallery.setOnClickListener { galleryLauncher.launch("image/*") }
        binding.btnRun.setOnClickListener     { runSegmentation() }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == R.id.action_about) { showAbout(); return true }
        return super.onOptionsItemSelected(item)
    }

    override fun onDestroy() {
        super.onDestroy()
        segmentor.close()
    }

    // ── Camera ─────────────────────────────────────────────────────────────

    private fun requestCameraOrLaunch() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) launchCamera()
        else cameraPermLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun launchCamera() {
        val file = File(cacheDir, "images").also { it.mkdirs() }
            .let { File(it, "capture.jpg") }
        cameraUri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        cameraLauncher.launch(cameraUri!!)
    }

    // ── Image loading ──────────────────────────────────────────────────────

    private fun setSourceImage(uri: Uri) {
        val bitmap = try {
            contentResolver.openInputStream(uri)?.use { stream ->
                // Sample down large images to avoid OOM (max 2048px)
                val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeStream(stream, null, opts)
                val scale = maxOf(opts.outWidth, opts.outHeight) / 2048
                val opts2 = BitmapFactory.Options().apply {
                    inSampleSize = if (scale > 1) scale else 1
                }
                contentResolver.openInputStream(uri)?.use { s2 ->
                    BitmapFactory.decodeStream(s2, null, opts2)
                }
            }
        } catch (e: Exception) {
            showSnack("Failed to load image: ${e.message}")
            null
        } ?: return

        currentBitmap = bitmap
        binding.ivOriginal.setImageBitmap(bitmap)
        binding.ivSegmented.setImageResource(R.drawable.ic_placeholder_pet)
    }

    // ── Inference ──────────────────────────────────────────────────────────

    private fun runSegmentation() {
        val bmp = currentBitmap ?: run {
            showSnack(getString(R.string.snack_pick_image))
            return
        }
        setUiBusy(true)

        lifecycleScope.launch {
            val overlay = withContext(Dispatchers.Default) {
                segmentor.renderOverlay(bmp, segmentor.segment(bmp))
            }
            binding.ivSegmented.setImageBitmap(overlay)
            setUiBusy(false)
        }
    }

    // ── UI helpers ─────────────────────────────────────────────────────────

    private fun setUiBusy(busy: Boolean) {
        binding.progressBar.visibility = if (busy) View.VISIBLE else View.GONE
        binding.btnRun.isEnabled     = !busy
        binding.btnCamera.isEnabled  = !busy
        binding.btnGallery.isEnabled = !busy
    }

    private fun showSnack(msg: String) =
        Snackbar.make(binding.root, msg, Snackbar.LENGTH_SHORT).show()

    private fun showAbout() {
        AlertDialog.Builder(this)
            .setTitle(R.string.about_title)
            .setMessage(R.string.about_message)
            .setPositiveButton(R.string.about_ok, null)
            .show()
    }
}
