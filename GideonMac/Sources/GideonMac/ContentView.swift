
import SwiftUI
import Vision

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var recognitionService = RecognitionService()
    
    var body: some View {
        ZStack {
            Color.black.edgesIgnoringSafeArea(.all)
            
            // Camera Feed (flipped horizontally to un-mirror)
            if let frame = cameraManager.currentFrame {
                CameraPreview(image: frame)
                    .scaleEffect(x: -1, y: 1)  // Un-mirror the front camera
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(2)
                    Text("Initializing Camera...")
                        .font(.title3)
                        .padding(.top)
                        .foregroundColor(.white.opacity(0.8))
                }
            }
            
            // HUD Interface
            VStack {
                // Header
                HStack {
                    VStack(alignment: .leading) {
                        Text("GIDEON")
                            .font(.system(size: 24, weight: .heavy, design: .monospaced))
                            .foregroundColor(.white)
                            .tracking(2)
                        Text("NEURAL INTERFACE")
                            .font(.caption)
                            .foregroundColor(.green.opacity(0.8))
                            .tracking(1)
                    }
                    Spacer()
                    FPSBadge(fps: cameraManager.fps)
                }
                .padding()
                .background(VisualEffectBlur(material: .hudWindow, blendingMode: .withinWindow))
                .cornerRadius(16)
                .padding()
                
                Spacer()
                
                // Bottom Control & Info Panel
                HStack(alignment: .bottom, spacing: 20) {
                    
                    // Finger Count Box - Using Vision framework
                    VStack {
                        Text("\(cameraManager.fingerCount)")
                            .font(.system(size: 72, weight: .bold, design: .rounded))
                            .foregroundColor(.white)
                            .shadow(color: .green.opacity(0.5), radius: 10, x: 0, y: 0)
                        Text("FINGERS")
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(.white.opacity(0.6))
                    }
                    .frame(width: 120, height: 140)
                    .background(VisualEffectBlur(material: .popover, blendingMode: .withinWindow))
                    .cornerRadius(20)
                    .overlay(
                        RoundedRectangle(cornerRadius: 20)
                            .stroke(Color.white.opacity(0.1), lineWidth: 1)
                    )
                    
                    // Prediction Box
                    if let prediction = recognitionService.prediction {
                        HStack(spacing: 30) {
                            // Model Input Preview
                            if let debugImage = recognitionService.lastInputImage {
                                VStack {
                                    Image(nsImage: debugImage)
                                        .resizable()
                                        .interpolation(.none)
                                        .frame(width: 60, height: 60)
                                        .cornerRadius(8)
                                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.2), lineWidth: 1))
                                    Text("INPUT")
                                        .font(.system(size: 10, weight: .bold))
                                        .foregroundColor(.white.opacity(0.5))
                                }
                            }
                            
                            Divider().background(Color.white.opacity(0.2))
                            
                            // Result
                            VStack(alignment: .leading) {
                                Text("DETECTED")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(.green)
                                
                                Text("\(prediction)")
                                    .font(.system(size: 60, weight: .heavy, design: .monospaced))
                                    .foregroundColor(.green)
                            }
                            
                            // Confidence
                            RingView(percentage: Double(recognitionService.confidence))
                                .frame(width: 50, height: 50)
                        }
                        .padding(20)
                        .background(VisualEffectBlur(material: .popover, blendingMode: .withinWindow))
                        .cornerRadius(20)
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(Color.green.opacity(0.3), lineWidth: 1)
                        )
                    }
                    
                    Spacer()
                }
                .padding()
            }
        }
        .onAppear {
            cameraManager.start()
        }
        .onChange(of: cameraManager.fingerCount) { oldValue, newValue in
            recognitionService.process(fingerCount: newValue)
        }
    }
}

// MARK: - Components

struct FPSBadge: View {
    let fps: Double
    
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(fps > 20 ? Color.green : Color.orange)
                .frame(width: 8, height: 8)
            Text("\(Int(fps)) FPS")
                .font(.system(.caption, design: .monospaced))
                .fontWeight(.bold)
                .foregroundColor(.white.opacity(0.8))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color.black.opacity(0.3))
        .cornerRadius(20)
    }
}

struct RingView: View {
    var percentage: Double
    
    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.white.opacity(0.1), lineWidth: 5)
            Circle()
                .trim(from: 0, to: CGFloat(percentage))
                .stroke(
                    LinearGradient(gradient: Gradient(colors: [.green, .mint]), startPoint: .top, endPoint: .bottom),
                    style: StrokeStyle(lineWidth: 5, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
            Text("\(Int(percentage * 100))%")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(.white)
        }
    }
}

struct VisualEffectBlur: NSViewRepresentable {
    var material: NSVisualEffectView.Material
    var blendingMode: NSVisualEffectView.BlendingMode
    
    func makeNSView(context: Context) -> NSVisualEffectView {
        let visualEffectView = NSVisualEffectView()
        visualEffectView.material = material
        visualEffectView.blendingMode = blendingMode
        visualEffectView.state = .active
        return visualEffectView
    }
    
    func updateNSView(_ visualEffectView: NSVisualEffectView, context: Context) {
        visualEffectView.material = material
        visualEffectView.blendingMode = blendingMode
    }
}

struct CameraPreview: NSViewRepresentable {
    let image: CGImage
    
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        view.layer = CALayer()
        view.layer?.contentsGravity = .resize
        view.wantsLayer = true
        return view
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {
        nsView.layer?.contents = image
    }
}

struct HandOverlay: View {
    let landmarks: [[CGPoint]]
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                for hand in landmarks {
                    // Draw connections
                    if hand.count >= 21 {
                        // Wrist to Thumb
                        addLines(path: &path, points: [hand[0], hand[1], hand[2], hand[3], hand[4]], geo: geometry)
                        // Wrist to Index
                        addLines(path: &path, points: [hand[0], hand[5], hand[6], hand[7], hand[8]], geo: geometry)
                        // Wrist to Middle
                        addLines(path: &path, points: [hand[0], hand[9], hand[10], hand[11], hand[12]], geo: geometry)
                        // Wrist to Ring
                        addLines(path: &path, points: [hand[0], hand[13], hand[14], hand[15], hand[16]], geo: geometry)
                        // Wrist to Little
                        addLines(path: &path, points: [hand[0], hand[17], hand[18], hand[19], hand[20]], geo: geometry)
                    }
                    
                    // Draw joints
                    for point in hand {
                        let x = point.x * geometry.size.width
                        let y = (1 - point.y) * geometry.size.height
                        path.addEllipse(in: CGRect(x: x - 4, y: y - 4, width: 8, height: 8))
                    }
                }
            }
            .stroke(Color.green, lineWidth: 2)
            .shadow(color: .green, radius: 5)
        }
    }
    
    private func addLines(path: inout Path, points: [CGPoint], geo: GeometryProxy) {
        guard let first = points.first else { return }
        let startX = first.x * geo.size.width
        let startY = (1 - first.y) * geo.size.height
        path.move(to: CGPoint(x: startX, y: startY))
        
        for i in 1..<points.count {
            let pt = points[i]
            let x = pt.x * geo.size.width
            let y = (1 - pt.y) * geo.size.height
            path.addLine(to: CGPoint(x: x, y: y))
        }
    }
}
