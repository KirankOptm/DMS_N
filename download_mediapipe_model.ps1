# Download MediaPipe Face Landmarker TFLite model directly
# MediaPipe usually provides standalone .tflite files

Write-Host "Downloading MediaPipe Face Landmarker TFLite model..."

# Try different model URLs from MediaPipe repository
$urls = @(
    # Float16 quantized model
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.tflite",
    # Float16 with blendshapes
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker_v2_with_blendshapes.tflite",
    # Try the model from the task file structure
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/model.tflite"
)

foreach ($url in $urls) {
    Write-Host "`nTrying: $url"
    $filename = Split-Path $url -Leaf
    
    try {
        Invoke-WebRequest -Uri $url -OutFile $filename -ErrorAction Stop
        $sizeMB = [math]::Round((Get-Item $filename).Length / 1MB, 2)
        Write-Host "Downloaded: $filename ($sizeMB MB)"
        
        # Check if it's a valid TFLite file
        $bytes = [System.IO.File]::ReadAllBytes($filename)
        $header = [System.Text.Encoding]::ASCII.GetString($bytes[0..3])
        if ($header -eq 'TFL3') {
            Write-Host "Valid TFLite model"
            Write-Host "`nNext: python check_tflite_model.py $filename"
            exit 0
        }
    }
    catch {
        Write-Host "Failed: $($_.Exception.Message)"
    }
}

Write-Host "`nAll download attempts failed"
Write-Host "Try checking MediaPipe GitHub for model links:"
Write-Host "https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_landmark"
