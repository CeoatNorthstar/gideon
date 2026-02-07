
import Foundation
import AppKit

/// Manages communication with Python hand tracking backend
@MainActor
class PythonBridge: ObservableObject {
    @Published var fingerCount: Int = 0
    @Published var isReady: Bool = false
    @Published var error: String?
    
    private var process: Process?
    private var stdin: FileHandle?
    private var stdout: FileHandle?
    private var readBuffer = Data()
    
    private var pythonPath: String = ""
    private var scriptPath: String = ""
    
    init() {
        // Find Python path - try venv first, then system python
        let fm = FileManager.default
        
        // Try to find project root (for development)
        let homeDir = fm.homeDirectoryForCurrentUser
        let projectVenv = homeDir.appendingPathComponent("Developer/gideon/.venv/bin/python3")
        let projectScript = homeDir.appendingPathComponent("Developer/gideon/GideonMac/hand_tracker_backend.py")
        
        // Check if running from app bundle
        if let bundleScript = Bundle.main.path(forResource: "hand_tracker_backend", ofType: "py") {
            self.scriptPath = bundleScript
        } else if fm.fileExists(atPath: projectScript.path) {
            self.scriptPath = projectScript.path
        }
        
        // Try venv python, then system python3
        if fm.fileExists(atPath: projectVenv.path) {
            self.pythonPath = projectVenv.path
        } else if let systemPython = findSystemPython() {
            self.pythonPath = systemPython
        } else {
            self.pythonPath = "/usr/bin/python3"
        }
        
        print("PythonBridge: Using python at \(pythonPath)")
        print("PythonBridge: Using script at \(scriptPath)")
    }
    
    private func findSystemPython() -> String? {
        // Try common Python locations
        let paths = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3"
        ]
        for path in paths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        return nil
    }
    
    func start() {
        guard process == nil else { return }
        
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonPath)
        proc.arguments = [scriptPath]
        proc.currentDirectoryURL = URL(fileURLWithPath: scriptPath).deletingLastPathComponent()
        
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        
        proc.standardInput = stdinPipe
        proc.standardOutput = stdoutPipe
        proc.standardError = stderrPipe
        
        stdin = stdinPipe.fileHandleForWriting
        stdout = stdoutPipe.fileHandleForReading
        
        // Read stdout asynchronously
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if !data.isEmpty {
                Task { @MainActor in
                    self?.handleOutput(data)
                }
            }
        }
        
        // Log stderr
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if let str = String(data: data, encoding: .utf8), !str.isEmpty {
                print("[Python stderr]: \(str)")
            }
        }
        
        do {
            try proc.run()
            process = proc
            print("Python backend started")
        } catch {
            self.error = "Failed to start Python: \(error.localizedDescription)"
            print(self.error!)
        }
    }
    
    func stop() {
        stdin?.closeFile()
        process?.terminate()
        process = nil
        isReady = false
    }
    
    func processFrame(_ cgImage: CGImage) {
        guard isReady, let stdin = stdin else { return }
        
        // Convert CGImage to JPEG and base64
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        guard let tiffData = nsImage.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let jpegData = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.7]) else {
            return
        }
        
        let base64 = jpegData.base64EncodedString()
        let message: [String: Any] = ["command": "process", "frame": base64]
        
        if let jsonData = try? JSONSerialization.data(withJSONObject: message),
           var jsonString = String(data: jsonData, encoding: .utf8) {
            jsonString += "\n"
            if let data = jsonString.data(using: .utf8) {
                stdin.write(data)
            }
        }
    }
    
    private func handleOutput(_ data: Data) {
        readBuffer.append(data)
        
        // Process complete lines
        while let newlineIndex = readBuffer.firstIndex(of: UInt8(ascii: "\n")) {
            let lineData = readBuffer[..<newlineIndex]
            readBuffer = readBuffer[(newlineIndex + 1)...]
            
            guard let line = String(data: lineData, encoding: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else {
                continue
            }
            
            if json["status"] as? String == "ready" {
                isReady = true
                print("Python backend ready")
            } else if let count = json["count"] as? Int {
                fingerCount = count
            } else if let err = json["error"] as? String {
                error = err
            }
        }
    }
}
