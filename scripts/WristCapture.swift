import Foundation
import AVFoundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers

// Select the AVCaptureDevice itself by identity; NEVER translate its list index to OpenCV.
final class Writer: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let destination: URL
    let maximumWidth: Double
    let context = CIContext(options: [.useSoftwareRenderer: false])
    var last = Date.distantPast
    init(_ destination: URL, maximumWidth: Double) { self.destination = destination; self.maximumWidth = maximumWidth }
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard Date().timeIntervalSince(last) > 0.10, let pixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        last = Date()
        let ci = CIImage(cvPixelBuffer: pixel)
        let scale = min(1.0, maximumWidth / ci.extent.width)
        let resized = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cg = context.createCGImage(resized, from: resized.extent) else { return }
        let data = NSMutableData()
        guard let target = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(target, cg, [kCGImageDestinationLossyCompressionQuality:0.94] as CFDictionary)
        guard CGImageDestinationFinalize(target) else { return }
        do { try (data as Data).write(to: destination, options: .atomic) } catch { fputs("Frame write failed\n", stderr) }
    }
}
let args = CommandLine.arguments
if args.count != 4 && args.count != 5 { fputs("Usage: wrist-capture exact-name exact-id output.jpg [960|1920]\n", stderr); exit(2) }
let maximumWidth = args.count == 5 ? Double(args[4]) ?? 0 : 960
guard maximumWidth == 960 || maximumWidth == 1920 else { fputs("Unsupported capture width\n",stderr);exit(2) }
let expectedName=args[1], expectedID=args[2]
guard !expectedName.lowercased().contains("realsense") else { fputs("Observer excluded\n",stderr);exit(2) }
let all = AVCaptureDevice.devices(for: .video)
let named=all.filter { $0.localizedName == expectedName && !$0.localizedName.lowercased().contains("realsense") }
let exact=named.filter { $0.uniqueID == expectedID }
let usbSuffix=String(expectedID.lowercased().suffix(8))
let byVendorProduct=named.filter { usbSuffix.count == 8 && $0.uniqueID.lowercased().hasSuffix(usbSuffix) }
let matches = !exact.isEmpty ? exact : (!byVendorProduct.isEmpty ? byVendorProduct : named)
guard matches.count == 1, let device = matches.first else { fputs("Wrist camera missing or ambiguous\n",stderr);exit(3) }
let state=AVCaptureDevice.authorizationStatus(for: .video)
if state == .notDetermined {
    let semaphore=DispatchSemaphore(value:0)
    AVCaptureDevice.requestAccess(for:.video) { ok in semaphore.signal() }
    _=semaphore.wait(timeout:.now()+30)
}
guard AVCaptureDevice.authorizationStatus(for:.video) == .authorized else { fputs("Camera permission required in macOS\n",stderr);exit(4) }
let session=AVCaptureSession()
session.beginConfiguration()
do {
 let input=try AVCaptureDeviceInput(device:device)
 guard session.canAddInput(input) else { exit(5) }
 session.addInput(input)
} catch { fputs("Cannot open wrist input\n",stderr);exit(5) }
guard session.canSetSessionPreset(.hd1920x1080) else { fputs("Required wrist format unavailable\n",stderr);exit(5) }
session.sessionPreset = .hd1920x1080
let output=AVCaptureVideoDataOutput()
output.alwaysDiscardsLateVideoFrames=true
output.videoSettings=[kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
let writer=Writer(URL(fileURLWithPath:args[3]), maximumWidth:maximumWidth)
output.setSampleBufferDelegate(writer, queue:DispatchQueue(label:"lerobot.wrist.capture"))
guard session.canAddOutput(output) else { exit(6) }
session.addOutput(output)
session.commitConfiguration()
fputs("Bound wrist: \(device.localizedName) [\(device.uniqueID)]\n",stderr)
session.startRunning()
RunLoop.main.run()
