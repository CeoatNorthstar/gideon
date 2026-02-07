#!/bin/bash

APP_NAME="Gideon"
BUILD_DIR=".build/release"
OUTPUT_DIR="."

echo "🚀 Building GideonMac..."
swift build -c release

if [ $? -ne 0 ]; then
    echo "❌ Build failed."
    exit 1
fi

echo "📦 Packaging as $APP_NAME.app..."

# Create App Bundle Structure
mkdir -p "$APP_NAME.app/Contents/MacOS"
mkdir -p "$APP_NAME.app/Contents/Resources"

# Copy Executable
cp "$BUILD_DIR/GideonMac" "$APP_NAME.app/Contents/MacOS/$APP_NAME"

# Copy Model (Must use -r for .mlpackage directory)
if [ -d "MNIST_CNN.mlpackage" ]; then
    echo "📄 Copying model to bundle resources..."
    cp -r "MNIST_CNN.mlpackage" "$APP_NAME.app/Contents/Resources/"
else
    echo "⚠️ Warning: MNIST_CNN.mlpackage not found in current directory. App may crash if model is missing."
fi

# Copy Python backend script
if [ -f "hand_tracker_backend.py" ]; then
    echo "🐍 Copying Python backend..."
    cp "hand_tracker_backend.py" "$APP_NAME.app/Contents/Resources/"
fi

# Create Info.plist
cat > "$APP_NAME.app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>com.gideon.mac</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>NSCameraUsageDescription</key>
    <string>We need access to the camera to detect hands and fingers.</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo "✅ $APP_NAME.app created successfully!"
echo "👉 Run with: open $APP_NAME.app"
