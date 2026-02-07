
import AVFoundation
import Vision
import CoreImage

class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    @Published var currentFrame: CGImage?
    @Published var handLandmarks: [[CGPoint]] = []
    @Published var fingerCount: Int = 0
    @Published var fps: Double = 0.0
    
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let context = CIContext()
    
    private var frameCount = 0
    private var fpsTimer = Date()
    
    // Enterprise-level: Temporal smoothing buffer
    private var recentCounts: [Int] = []
    private let smoothingWindowSize = 5  // Use mode of last 5 readings
    
    private let handPoseRequest = VNDetectHumanHandPoseRequest()
    
    override init() {
        super.init()
        setupSession()
        handPoseRequest.maximumHandCount = 2
    }
    
    func start() {
        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()
        }
    }
    
    private func setupSession() {
        session.sessionPreset = .hd1280x720
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else { return }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
            }
            
            if session.canAddOutput(output) {
                session.addOutput(output)
                output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            }
        } catch {
            print("Failed to setup camera: \(error)")
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Hand Tracking
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([handPoseRequest])
            if let observations = handPoseRequest.results, !observations.isEmpty {
                processHandObservations(observations)
            } else {
                updateSmoothedCount(0)
                DispatchQueue.main.async {
                    self.handLandmarks = []
                }
            }
        } catch {
            print("Hand pose detection failed: \(error)")
        }
        
        // Update Frame
        if let cgImage = self.context.createCGImage(ciImage, from: ciImage.extent) {
            DispatchQueue.main.async {
                self.currentFrame = cgImage
                self.updateFPS()
            }
        }
    }
    
    private func processHandObservations(_ observations: [VNHumanHandPoseObservation]) {
        var allLandmarks: [[CGPoint]] = []
        var totalFingers = 0
        
        for observation in observations {
            var points: [CGPoint] = []
            guard let recognizedPoints = try? observation.recognizedPoints(.all) else { continue }
            
            let keys: [VNHumanHandPoseObservation.JointName] = [
                .wrist,
                .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
                .indexMCP, .indexPIP, .indexDIP, .indexTip,
                .middleMCP, .middlePIP, .middleDIP, .middleTip,
                .ringMCP, .ringPIP, .ringDIP, .ringTip,
                .littleMCP, .littlePIP, .littleDIP, .littleTip
            ]
            
            var highConfidenceCount = 0
            for key in keys {
                if let point = recognizedPoints[key], point.confidence > 0.5 {
                    points.append(point.location)
                    if point.confidence > 0.7 { highConfidenceCount += 1 }
                } else {
                    points.append(.zero)
                }
            }
            
            // Enterprise: Require at least 15 joints with high confidence
            let validPoints = points.filter { $0 != .zero }
            if validPoints.count > 15 && highConfidenceCount > 10 {
                allLandmarks.append(points)
                totalFingers += countFingers(observation)
            }
        }
        
        updateSmoothedCount(totalFingers)
        DispatchQueue.main.async {
            self.handLandmarks = allLandmarks
        }
    }
    
    private func updateSmoothedCount(_ rawCount: Int) {
        // Add to buffer
        recentCounts.append(rawCount)
        if recentCounts.count > smoothingWindowSize {
            recentCounts.removeFirst()
        }
        
        // Calculate mode (most frequent value) for stability
        let smoothedCount = calculateMode(recentCounts)
        
        DispatchQueue.main.async {
            self.fingerCount = smoothedCount
        }
    }
    
    private func calculateMode(_ values: [Int]) -> Int {
        var frequency: [Int: Int] = [:]
        for value in values {
            frequency[value, default: 0] += 1
        }
        return frequency.max(by: { $0.value < $1.value })?.key ?? 0
    }
    
    private func countFingers(_ observation: VNHumanHandPoseObservation) -> Int {
        var count = 0
        guard let points = try? observation.recognizedPoints(.all) else { return 0 }
        guard let wrist = points[.wrist], wrist.confidence > 0.3 else { return 0 }
        
        // EXACTLY MATCHING live.py LOGIC:
        // FINGER_TIPS = [4, 8, 12, 16, 20]  -> thumbTip, indexTip, middleTip, ringTip, littleTip
        // FINGER_PIPS = [2, 6, 10, 14, 18]  -> thumbMCP, indexPIP, middlePIP, ringPIP, littlePIP
        
        // THUMB: Compare X-axis distance from wrist
        // thumb_tip_dist = abs(thumb_tip[0] - wrist[0])
        // thumb_base_dist = abs(thumb_base[0] - wrist[0])  # Uses index 2 = thumbIP in Swift
        // if thumb_tip_dist > thumb_base_dist: finger_count += 1
        if let thumbTip = points[.thumbTip], let thumbIP = points[.thumbIP],
           thumbTip.confidence > 0.3, thumbIP.confidence > 0.3 {
            let thumbTipDist = abs(thumbTip.location.x - wrist.location.x)
            let thumbBaseDist = abs(thumbIP.location.x - wrist.location.x)
            if thumbTipDist > thumbBaseDist {
                count += 1
            }
        }
        
        // OTHER 4 FINGERS: Compare Y-axis (tip vs PIP)
        // In Vision framework: higher Y = up (opposite of image coords)
        // In live.py: if tip_y < pip_y (lower Y = higher in image)
        // So in Vision: if tip.location.y > pip.location.y (higher = up)
        let fingerJoints: [(tip: VNHumanHandPoseObservation.JointName, pip: VNHumanHandPoseObservation.JointName)] = [
            (.indexTip, .indexPIP),
            (.middleTip, .middlePIP),
            (.ringTip, .ringPIP),
            (.littleTip, .littlePIP)
        ]
        
        for (tipJoint, pipJoint) in fingerJoints {
            guard let tip = points[tipJoint], let pip = points[pipJoint],
                  tip.confidence > 0.3, pip.confidence > 0.3 else { continue }
            
            // Finger is extended if tip is higher than PIP
            // Vision: higher Y = up, so tip.y > pip.y means extended
            if tip.location.y > pip.location.y {
                count += 1
            }
        }
        
        return count
    }
    
    private func updateFPS() {
        frameCount += 1
        let now = Date()
        let interval = now.timeIntervalSince(fpsTimer)
        if interval >= 1.0 {
            fps = Double(frameCount) / interval
            frameCount = 0
            fpsTimer = now
        }
    }
}
