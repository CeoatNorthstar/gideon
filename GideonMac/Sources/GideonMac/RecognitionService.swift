
import CoreML
import Vision
import AppKit

@MainActor
class RecognitionService: ObservableObject {
    @Published var prediction: Int?
    @Published var confidence: Float = 0.0
    @Published var lastInputImage: NSImage?
    
    private var model: MLModel?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        // Load model from bundle resources
        // We assume MNIST_CNN.mlpackage is copied to bundle resources
        // OR we load from a known path if it's a command line tool/dev build
        
        // For development loop, we can try to find it in the module bundle or main bundle.
        // SwiftPM resources: Bundle.module
        
        // However, referencing Bundle.module requires 'resources' in Package.swift.
        // We didn't add it yet.
        // Let's assume we can load it from a relative path for now or try Bundle.main.
        
        // Try to find model in Bundle first, then local path
        let bundleURL = Bundle.main.url(forResource: "MNIST_CNN", withExtension: "mlpackage")
        let localURL = URL(fileURLWithPath: "MNIST_CNN.mlpackage")
        
        var modelToLoad: URL?
        
        if let bundleURL = bundleURL {
            modelToLoad = bundleURL
        } else if FileManager.default.fileExists(atPath: localURL.path) {
            print("Found model at local path: \(localURL.path)")
            modelToLoad = localURL
        }
        
        guard let modelURL = modelToLoad else {
            print("Error: Could not find MNIST_CNN.mlpackage in Bundle or current directory.")
            return
        }
        
        do {
            print("Loading model from: \(modelURL.lastPathComponent)...")
            let compiledUrl = try MLModel.compileModel(at: modelURL)
            self.model = try MLModel(contentsOf: compiledUrl)
            print("✓ Model loaded successfully.")
        } catch {
            print("✗ Failed to load/compile model: \(error)")
        }
    }
    
    func process(fingerCount: Int) {
        let digit = fingerCount % 10
        
        guard let cgImage = DigitRenderer.shared.render(digit: digit) else { return }
        self.lastInputImage = NSImage(cgImage: cgImage, size: NSSize(width: 28, height: 28))
        
        guard let model = model else { 
            // Try to load again if missed
            loadModel()
            return 
        }
        
        guard let inputMultiArray = try? MLMultiArray(shape: [1, 1, 28, 28], dataType: .float32) else { return }
        
        let width = 28
        let height = 28
        let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
        
        let mean: Float = 0.1307
        let std: Float = 0.3081
        
        // Populate MultiArray
        for y in 0..<height {
            for x in 0..<width {
                let color = bitmapRep.colorAt(x: x, y: y) ?? .black
                let gray = Float(color.whiteComponent)
                let normalized = (gray - mean) / std
                
                let index = [0, 0, NSNumber(value: y), NSNumber(value: x)] as [NSNumber]
                inputMultiArray[index] = NSNumber(value: normalized)
            }
        }
        
        // Predict
        do {
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["input_1": inputMultiArray])
            let output = try model.prediction(from: inputProvider)
            
            guard let outputArray = output.featureValue(for: "output_1")?.multiArrayValue else { return }
            
            // Output processing
            var maxVal: Float = -Float.infinity
            var maxIdx = -1
            
            let pointer = outputArray.dataPointer.bindMemory(to: Float.self, capacity: 10)
            
            var exps: [Float] = []
            var expSum: Float = 0
            
            for i in 0..<10 {
                let val = pointer[i]
                if val > maxVal {
                    maxVal = val
                    maxIdx = i
                }
                let e = exp(val)
                exps.append(e)
                expSum += e
            }
            
            let conf = exps[maxIdx] / expSum
            
            DispatchQueue.main.async {
                self.prediction = maxIdx
                self.confidence = conf
            }
            
        } catch {
            print("Prediction failed: \(error)")
        }
    }
}
