package com.lithing.tracking

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import android.view.WindowManager
import androidx.core.content.ContextCompat
import app.tauri.PermissionState
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.Permission
import app.tauri.annotation.PermissionCallback
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSArray
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/** Rust (lithing-tauri-plugin, src/android.rs). */
object LithingNative {
  /** Frame accepted. */
  const val PUSHED = 0
  /** No tracker yet (the models are still loading). */
  const val NOT_READY = 1
  /** The tracker is gone: close the camera. */
  const val CLOSED = 2

  /**
   * One YUV_420_888 image: converted to RGB, rotated clockwise by [rotation] degrees and
   * handed to the tracker before this returns (the buffers may be released afterwards).
   */
  @JvmStatic
  external fun pushFrame(
    y: ByteBuffer, u: ByteBuffer, v: ByteBuffer,
    yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
    width: Int, height: Int, rotation: Int
  ): Int
}

@InvokeArg
class StartCameraArgs {
  /** Position in listCameras (front cameras first, so 0 is the selfie camera). */
  var index: Int = 0
  var width: Int = 640
  var height: Int = 480
  var fps: Int = 30
  /** Hold the rate in low light (a fixed AE range) instead of letting exposure lower it. */
  var keepFrameRate: Boolean = true
}

/**
 * Native tracking's camera on Android: Camera2 delivering YUV_420_888 images to Rust
 * (no preview surface, no copy in Kotlin), and the models copied out of the APK.
 * The camera follows the activity: closed while the app is in the background,
 * reopened when it comes back, so it is never held where the user cannot see the app.
 */
@TauriPlugin(permissions = [Permission(strings = [Manifest.permission.CAMERA], alias = "camera")])
class LithingCameraPlugin(private val activity: Activity) : Plugin(activity) {
  private val manager by lazy { activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager }
  private val thread = HandlerThread("lithing-camera").apply { start() }
  /** Every camera state change happens on this thread. */
  private val handler = Handler(thread.looper)

  private class Selected(val id: String, val label: String, val size: Size, val fps: Range<Int>, val sensor: Int, val front: Boolean)

  private var selected: Selected? = null
  private var device: CameraDevice? = null
  private var session: CameraCaptureSession? = null
  private var reader: ImageReader? = null
  /** Bumped on every open / close: callbacks of an older camera are ignored. */
  private var generation = 0
  @Volatile private var paused = false
  @Volatile private var displayRotation = 0
  private var notReadySince = 0L

  // ---- Commands ------------------------------------------------------------------------

  @Command
  fun listCameras(invoke: Invoke) {
    try {
      val list = JSArray()
      orderedCameras().forEachIndexed { index, (_, label) -> list.put("$index: $label (Camera2)") }
      val result = JSObject()
      result.put("cameras", list)
      invoke.resolve(result)
    } catch (ex: Exception) {
      invoke.reject("Could not list the cameras: ${ex.message}")
    }
  }

  @Command
  fun startCamera(invoke: Invoke) {
    if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
      handler.post { open(invoke) }
    } else {
      requestPermissionForAlias("camera", invoke, "cameraPermission")
    }
  }

  @PermissionCallback
  fun cameraPermission(invoke: Invoke) {
    if (getPermissionState("camera") == PermissionState.GRANTED) handler.post { open(invoke) }
    else invoke.reject("Camera permission denied")
  }

  @Command
  fun stopCamera(invoke: Invoke) {
    handler.post {
      selected = null
      close()
      invoke.resolve()
    }
  }

  /**
   * Copies assets/models/\*.onnx to the app's files folder (ONNX Runtime reads files) on the
   * first run after an install or update, and returns that folder.
   */
  @Command
  fun prepareModels(invoke: Invoke) {
    Thread {
      try {
        val names = activity.assets.list("models")?.filter { it.endsWith(".onnx") }.orEmpty()
        if (names.isEmpty()) {
          invoke.reject("This build has no tracking models (assets/models)")
          return@Thread
        }
        val dir = File(activity.filesDir, "lithing-models")
        val stamp = File(dir, ".installed")
        @Suppress("DEPRECATION")
        val installed = activity.packageManager.getPackageInfo(activity.packageName, 0).lastUpdateTime.toString()
        val current = stamp.isFile && stamp.readText() == installed && names.all { File(dir, it).isFile }
        if (!current) {
          dir.mkdirs()
          stamp.delete()
          for (name in names) {
            val part = File(dir, "$name.part")
            activity.assets.open("models/$name").use { input -> part.outputStream().use { input.copyTo(it, 1 shl 16) } }
            val target = File(dir, name)
            target.delete()
            if (!part.renameTo(target)) throw IllegalStateException("could not write $target")
          }
          stamp.writeText(installed)
        }
        val result = JSObject()
        result.put("dir", dir.absolutePath)
        invoke.resolve(result)
      } catch (ex: Exception) {
        invoke.reject("Could not prepare the tracking models: ${ex.message}")
      }
    }.start()
  }

  // ---- Activity ------------------------------------------------------------------------

  override fun onPause() {
    paused = true
    handler.post { if (selected != null) close() }
  }

  override fun onResume() {
    paused = false
    handler.post { if (selected != null && device == null) open(null) }
  }

  override fun onDestroy() {
    handler.post {
      selected = null
      close()
      thread.quitSafely()
    }
  }

  // ---- Camera --------------------------------------------------------------------------

  /** Front cameras first, then back, then external; labels as the WebView names them. */
  private fun orderedCameras(): List<Pair<String, String>> {
    fun facing(id: String) = manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING)
    val rank = mapOf(CameraCharacteristics.LENS_FACING_FRONT to 0, CameraCharacteristics.LENS_FACING_BACK to 1)
    return manager.cameraIdList.toList()
      .sortedWith(compareBy({ rank[facing(it)] ?: 2 }, { it.toIntOrNull() ?: Int.MAX_VALUE }))
      .map { id ->
        val name = when (facing(id)) {
          CameraCharacteristics.LENS_FACING_FRONT -> "front"
          CameraCharacteristics.LENS_FACING_BACK -> "back"
          else -> "external"
        }
        id to "camera2 $id, facing $name"
      }
  }

  private fun select(args: StartCameraArgs): Selected {
    val cameras = orderedCameras()
    if (cameras.isEmpty()) throw IllegalStateException("This device has no camera")
    val (id, label) = cameras.getOrElse(args.index) { cameras[0] }
    val chars = manager.getCameraCharacteristics(id)
    val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: throw IllegalStateException("Camera $id lists no formats")
    val sizes = map.getOutputSizes(ImageFormat.YUV_420_888) ?: throw IllegalStateException("Camera $id has no YUV output")
    // The landscape size closest to the request (sensors list landscape sizes; rotation comes later).
    val (wantW, wantH) = if (args.width >= args.height) args.width to args.height else args.height to args.width
    val size = sizes.minByOrNull { Math.abs(it.width - wantW) + Math.abs(it.height - wantH) } ?: throw IllegalStateException("Camera $id has no sizes")
    val wantFps = args.fps.coerceIn(15, 60)
    val ranges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()
    // The highest rate up to the one asked for. Keep frame rate: the range with the highest
    // floor (fixed 30-30 if offered), so low light darkens the image instead of halving the
    // rate; otherwise the lowest floor, letting auto-exposure trade rate for brightness.
    val usable = ranges.filter { it.upper <= wantFps }
    val top = usable.maxOfOrNull { it.upper }
    val fps = usable.filter { it.upper == top }
      .let { candidates -> if (args.keepFrameRate) candidates.maxByOrNull { it.lower } else candidates.minByOrNull { it.lower } }
      ?: ranges.minByOrNull { it.upper } ?: Range(15, 30)
    val sensor = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
    val front = chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT
    return Selected(id, label, size, fps, sensor, front)
  }

  /** Clockwise rotation that makes the image upright on the screen as it is held now. */
  private fun rotation(camera: Selected): Int {
    val display = displayRotation
    return if (camera.front) (camera.sensor + display) % 360 else (camera.sensor - display + 360) % 360
  }

  private fun readDisplayRotation() {
    @Suppress("DEPRECATION")
    val surface = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) activity.display?.rotation ?: Surface.ROTATION_0
    else (activity.getSystemService(Context.WINDOW_SERVICE) as WindowManager).defaultDisplay.rotation
    displayRotation = when (surface) {
      Surface.ROTATION_90 -> 90
      Surface.ROTATION_180 -> 180
      Surface.ROTATION_270 -> 270
      else -> 0
    }
  }

  /** Opens the camera (`invoke` = a start command to answer, null = reopening after a pause). Camera thread. */
  @Suppress("MissingPermission")
  private fun open(invoke: Invoke?) {
    close()
    val camera = try {
      if (invoke != null) select(invoke.parseArgs(StartCameraArgs::class.java)) else selected ?: return
    } catch (ex: Exception) {
      invoke?.reject(ex.message ?: "Could not choose a camera")
      return
    }
    selected = camera
    if (paused) {
      // Opened when the activity resumes; the start is answered with the size it will have.
      invoke?.let { answer(it, camera) }
      return
    }
    val mine = ++generation
    val answered = AtomicBoolean(invoke == null)
    fun fail(message: String) {
      Log.w(TAG, message)
      if (answered.compareAndSet(false, true)) {
        selected = null
        invoke?.reject(message)
      }
      if (mine == generation) close()
    }
    readDisplayRotation()
    val imageReader = ImageReader.newInstance(camera.size.width, camera.size.height, ImageFormat.YUV_420_888, 3)
    reader = imageReader
    notReadySince = 0L
    var frames = 0
    imageReader.setOnImageAvailableListener({ source ->
      val image = try { source.acquireLatestImage() } catch (_: IllegalStateException) { null } ?: return@setOnImageAvailableListener
      try {
        if (mine != generation) return@setOnImageAvailableListener
        // The screen may have turned; reading it every 15 frames is enough.
        if (frames++ % 15 == 0) readDisplayRotation()
        val planes = image.planes
        val result = LithingNative.pushFrame(
          planes[0].buffer, planes[1].buffer, planes[2].buffer,
          planes[0].rowStride, planes[1].rowStride, planes[1].pixelStride,
          image.width, image.height, rotation(camera)
        )
        when (result) {
          LithingNative.CLOSED -> { selected = null; close() }
          LithingNative.NOT_READY -> {
            // Models load in a few seconds; a tracker that never comes must not keep the camera.
            val now = SystemClock.elapsedRealtime()
            if (notReadySince == 0L) notReadySince = now
            else if (now - notReadySince > 30_000) { Log.w(TAG, "no tracker for 30 s, closing the camera"); selected = null; close() }
          }
          else -> notReadySince = 0L
        }
      } finally {
        image.close()
      }
    }, handler)
    try {
      manager.openCamera(camera.id, object : CameraDevice.StateCallback() {
        override fun onOpened(opened: CameraDevice) {
          if (mine != generation) { opened.close(); return }
          device = opened
          try {
            @Suppress("DEPRECATION")
            opened.createCaptureSession(listOf(imageReader.surface), object : CameraCaptureSession.StateCallback() {
              override fun onConfigured(configured: CameraCaptureSession) {
                if (mine != generation) { configured.close(); return }
                session = configured
                try {
                  val request = opened.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                    addTarget(imageReader.surface)
                    set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                    set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, camera.fps)
                  }
                  configured.setRepeatingRequest(request.build(), null, handler)
                  if (answered.compareAndSet(false, true)) invoke?.let { answer(it, camera) }
                } catch (ex: Exception) {
                  fail("Could not start the camera: ${ex.message}")
                }
              }

              override fun onConfigureFailed(failed: CameraCaptureSession) {
                fail("The camera refused ${camera.size.width}x${camera.size.height} YUV")
              }
            }, handler)
          } catch (ex: Exception) {
            fail("Could not configure the camera: ${ex.message}")
          }
        }

        override fun onDisconnected(lost: CameraDevice) {
          // Another user took the camera (another app, or the WebView): it is theirs now,
          // so no reopening on resume. The tracker stays and waits; the page restarts it.
          lost.close()
          if (mine == generation) { device = null; selected = null; fail("The camera was disconnected") }
        }

        override fun onError(failed: CameraDevice, error: Int) {
          failed.close()
          if (mine == generation) { device = null; fail("Camera error $error") }
        }
      }, handler)
    } catch (ex: Exception) {
      fail("Could not open the camera: ${ex.message}")
    }
  }

  private fun answer(invoke: Invoke, camera: Selected) {
    val quarter = rotation(camera) % 180 == 90
    val result = JSObject()
    result.put("width", if (quarter) camera.size.height else camera.size.width)
    result.put("height", if (quarter) camera.size.width else camera.size.height)
    result.put("fps", camera.fps.upper)
    result.put("label", camera.label)
    invoke.resolve(result)
  }

  /** Releases the camera (camera thread). `selected` stays for a reopen after a pause. */
  private fun close() {
    generation++
    try { session?.close() } catch (_: Exception) {}
    try { device?.close() } catch (_: Exception) {}
    try { reader?.close() } catch (_: Exception) {}
    session = null
    device = null
    reader = null
  }

  companion object {
    private const val TAG = "LithingCamera"
  }
}
