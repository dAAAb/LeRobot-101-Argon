import Foundation
import AVFoundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers

// Select the AVCaptureDevice itself by identity; NEVER translate its list index to OpenCV.
final class Writer: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let destination: URL
    let device: AVCaptureDevice
    let expectedSize: (width: Int, height: Int)?
    let context = CIContext(options: [.useSoftwareRenderer: false])
    private let formatLock = NSLock()
    private var nativeConfigurationReady = false
    private var nativeFormatAccepted = false
    private var nativeDeadline: TimeInterval?
    var last = Date.distantPast
    var loggedFormat = false
    var loggedTransition = false
    init(_ destination: URL, device: AVCaptureDevice, expectedSize: (width: Int, height: Int)? = nil) {
        self.destination = destination; self.device = device; self.expectedSize = expectedSize
    }
    func enableNativeValidation() {
        formatLock.lock()
        nativeConfigurationReady = true
        nativeDeadline = ProcessInfo.processInfo.systemUptime + 3.0
        formatLock.unlock()
    }
    func requireNegotiatedFormat() {
        formatLock.lock()
        let accepted = nativeFormatAccepted
        formatLock.unlock()
        if !accepted {
            fputs("Observer native format did not deliver a matching frame within 3 seconds\n", stderr); exit(7)
        }
    }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard Date().timeIntervalSince(last) > 0.10, let pixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        formatLock.lock()
        let ready = nativeConfigurationReady, accepted = nativeFormatAccepted, deadline = nativeDeadline
        formatLock.unlock()
        // startRunning can reset a macOS UVC mode. Ignore startup buffers until
        // the host has reapplied the exact selected format to the running device.
        if expectedSize != nil && !ready { return }
        if expectedSize != nil && !accepted, let deadline = deadline,
           ProcessInfo.processInfo.systemUptime >= deadline {
            fputs("Observer native negotiation exceeded 3 seconds; no image published\n", stderr); exit(7)
        }
        let deliveredWidth = CVPixelBufferGetWidth(pixel), deliveredHeight = CVPixelBufferGetHeight(pixel)
        let source = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        if let size = expectedSize,
           deliveredWidth != size.width || deliveredHeight != size.height ||
           Int(source.width) != size.width || Int(source.height) != size.height {
            if !accepted, let deadline = deadline, ProcessInfo.processInfo.systemUptime < deadline {
                if !loggedTransition {
                    fputs("Observer negotiating: requested \(size.width)x\(size.height), delivered \(deliveredWidth)x\(deliveredHeight), device active \(source.width)x\(source.height); discarding transitional frames\n", stderr)
                    loggedTransition = true
                }
                return
            }
            fputs("Observer format mismatch: requested \(size.width)x\(size.height), delivered \(deliveredWidth)x\(deliveredHeight), device active \(source.width)x\(source.height); refusing resize or fallback\n", stderr); exit(7)
        }
        if expectedSize != nil {
            formatLock.lock(); nativeFormatAccepted = true; formatLock.unlock()
        }
        if !loggedFormat {
            fputs("Observer accepted frame: delivered \(deliveredWidth)x\(deliveredHeight), device active \(source.width)x\(source.height), pixel format \(CVPixelBufferGetPixelFormatType(pixel))\n", stderr)
            loggedFormat = true
        }
        last = Date()
        let ci = CIImage(cvPixelBuffer: pixel)
        let scale = min(1.0, 960.0 / ci.extent.width)
        let resized = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cg = context.createCGImage(resized, from: resized.extent) else { return }
        let data = NSMutableData()
        guard let target = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(target, cg, [kCGImageDestinationLossyCompressionQuality:0.86] as CFDictionary)
        guard CGImageDestinationFinalize(target) else { return }
        do { try (data as Data).write(to: destination, options: .atomic) } catch { fputs("Frame write failed\n", stderr) }
    }
}
let args = CommandLine.arguments
if args.count != 4 && args.count != 7 {
    fputs("Usage: observer-capture exact-name exact-id output.jpg [width height fps]\n", stderr); exit(2)
}
var requested: (width: Int, height: Int, fps: Double)?
if args.count == 7 {
    guard let width = Int(args[4]), let height = Int(args[5]), let fps = Double(args[6]),
          (1...8192).contains(width), (1...8192).contains(height), fps.isFinite, fps > 0, fps <= 240 else {
        fputs("Invalid observer width, height, or frame rate\n", stderr); exit(2)
    }
    requested = (width, height, fps)
}
let expectedName=args[1], expectedID=args[2]
guard expectedName.lowercased().contains("realsense") else { fputs("Production cameras excluded\n",stderr);exit(2) }
let all = AVCaptureDevice.devices(for: .video)
let named=all.filter { $0.localizedName == expectedName && $0.localizedName.lowercased().contains("realsense") }
let exact=named.filter { $0.uniqueID == expectedID }
let matches=exact
guard matches.count == 1, let device = matches.first else { fputs("Observer camera missing or ambiguous\n",stderr);exit(3) }
var selected: (format: AVCaptureDevice.Format, duration: CMTime)?
if let request = requested {
    for format in device.formats {
        let size = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        guard Int(size.width) == request.width && Int(size.height) == request.height else { continue }
        // UVC rates such as 6.000006 fps represent the requested 6 fps mode.
        // Use the device's exact endpoint duration, never an unsupported one.
        guard let range = format.videoSupportedFrameRateRanges.first(where: {
            request.fps >= $0.minFrameRate - 0.001 && request.fps <= $0.maxFrameRate + 0.001
        }) else { continue }
        let duration: CMTime
        if request.fps <= range.minFrameRate { duration = range.maxFrameDuration }
        else if request.fps >= range.maxFrameRate { duration = range.minFrameDuration }
        else { duration = CMTime(seconds: 1.0 / request.fps, preferredTimescale: 1_000_000_000) }
        selected = (format, duration); break
    }
    guard selected != nil else {
        fputs("Requested observer format is unavailable on this exact device\n", stderr); exit(7)
    }
}
let state=AVCaptureDevice.authorizationStatus(for: .video)
if state == .notDetermined {
    let semaphore=DispatchSemaphore(value:0)
    AVCaptureDevice.requestAccess(for:.video) { ok in semaphore.signal() }
    _=semaphore.wait(timeout:.now()+30)
}
guard AVCaptureDevice.authorizationStatus(for:.video) == .authorized else { fputs("Camera permission required in macOS\n",stderr);exit(4) }
let session=AVCaptureSession()
session.beginConfiguration()
if requested == nil && session.canSetSessionPreset(.vga640x480) { session.sessionPreset = .vga640x480 }
do {
 let input=try AVCaptureDeviceInput(device:device)
 guard session.canAddInput(input) else { exit(5) }
 session.addInput(input)
} catch { fputs("Cannot open observer input\n",stderr);exit(5) }
if let request = requested {
    let preset: AVCaptureSession.Preset?
    switch (request.width, request.height) {
    case (640, 480): preset = .vga640x480
    case (1280, 720): preset = .hd1280x720
    case (1920, 1080): preset = .hd1920x1080
    default: preset = nil
    }
    if let preset = preset {
        guard session.canSetSessionPreset(preset) else {
            fputs("Session cannot preserve the requested observer preset\n", stderr); exit(7)
        }
        session.sessionPreset = preset
    }
}
let output=AVCaptureVideoDataOutput()
output.alwaysDiscardsLateVideoFrames=true
// The macOS SDK specifies an empty dictionary for device-native samples.
// Avoid preset-sized BGRA conversion; CoreImage consumes the native YUV buffer.
output.videoSettings = requested == nil ? [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA] : [:]
let writer=Writer(URL(fileURLWithPath:args[3]), device: device, expectedSize: requested.map { ($0.width, $0.height) })
output.setSampleBufferDelegate(writer, queue:DispatchQueue(label:"lerobot.engineering.observer"))
guard session.canAddOutput(output) else { exit(6) }
session.addOutput(output)
if let choice = selected {
    // macOS has no inputPriority preset. Configure the native device format
    // and durations together, after adding input/output, before session commit.
    do {
        try device.lockForConfiguration()
        device.activeFormat = choice.format
        device.activeVideoMinFrameDuration = choice.duration
        device.activeVideoMaxFrameDuration = choice.duration
        device.unlockForConfiguration()
    } catch { fputs("Cannot configure the requested observer format\n", stderr); exit(7) }
}
session.commitConfiguration()
if let request = requested {
    let actual = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
    guard Int(actual.width) == request.width && Int(actual.height) == request.height else {
        fputs("Observer native format changed during configuration\n", stderr); exit(7)
    }
    fputs("Observer native format: \(actual.width)x\(actual.height), \(1.0 / CMTimeGetSeconds(device.activeVideoMinFrameDuration)) fps; aspect-preserving output <=960 px\n", stderr)
}
fputs("Engineering observer ONLY: \(device.localizedName) [\(device.uniqueID)]\n",stderr)
session.startRunning()
if let choice = selected {
    // Reapply once to the running device, not by scaling the output buffers.
    // The writer cannot publish until this configuration step has finished.
    do {
        try device.lockForConfiguration()
        device.activeFormat = choice.format
        device.activeVideoMinFrameDuration = choice.duration
        device.activeVideoMaxFrameDuration = choice.duration
        device.unlockForConfiguration()
    } catch { fputs("Cannot renegotiate the running observer format\n", stderr); exit(7) }
    let actual = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
    fputs("Observer running format reapplied: \(actual.width)x\(actual.height), \(1.0 / CMTimeGetSeconds(device.activeVideoMinFrameDuration)) fps\n", stderr)
    writer.enableNativeValidation()
    Timer.scheduledTimer(withTimeInterval: 3.0, repeats: false) { _ in writer.requireNegotiatedFormat() }
}
RunLoop.main.run()
