
import CoreGraphics
import CoreImage
import AppKit

@MainActor
class DigitRenderer {
    static let shared = DigitRenderer()
    
    func render(digit: Int) -> CGImage? {
        let size = CGSize(width: 28, height: 28)
        let canvas = NSImage(size: size)
        
        canvas.lockFocus()
        
        // Black background
        NSColor.black.setFill()
        let description_rect = NSRect(origin: .zero, size: size)
        description_rect.fill()
        
        // Draw digit centered
        let text = "\(digit)"
        let font = NSFont.systemFont(ofSize: 20, weight: .bold) // Adjust size to fill frame
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: NSColor.white
        ]
        
        let string = NSAttributedString(string: text, attributes: attributes)
        let textSize = string.size()
        let textRect = NSRect(
            x: (size.width - textSize.width) / 2,
            y: (size.height - textSize.height) / 2,
            width: textSize.width,
            height: textSize.height
        )
        
        string.draw(in: textRect)
        
        canvas.unlockFocus()
        
        return canvas.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }
}
