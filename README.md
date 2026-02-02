# 🎯 Circle Detection Web Application

A web-based TIFF image analyzer with circle detection and grid analysis, powered by **PyScript** - running Python directly in your browser without any backend server!

## ✨ Features

- 📁 **Drag & Drop Interface**: Easy file upload with drag-and-drop support
- 🔧 **Adjustable Parameters**: Fine-tune detection settings in real-time
- 🎨 **Visual Results**: View detected circles and threshold masks
- 📊 **Detailed Statistics**: Get comprehensive analysis data
- 🔲 **Grid Detection**: Automatic 4x4 grid position calculation
- 🚀 **No Backend Required**: Pure client-side processing with PyScript
- 💻 **Cross-Platform**: Works on any modern web browser

## 🛠️ Technology Stack

- **PyScript**: Python runtime in the browser
- **OpenCV**: Image processing and circle detection
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Pillow**: Image handling
- **HTML5/CSS3**: Modern responsive UI

## 📋 Requirements

- A modern web browser (Chrome, Firefox, Edge, Safari)
- Internet connection (for initial PyScript library loading)
- No Python installation required!

## 🚀 Quick Start

### Method 1: Direct Open (Recommended)

1. Simply open `index.html` in your web browser
2. Wait for PyScript to initialize (you'll see a loading indicator)
3. Upload your TIFF file
4. Adjust parameters if needed
5. Click "Process Image"
6. View your results!

### Method 2: Local Server (Recommended for Development)

Using the included `run.py` script:

```bash
# Run with default port 8000
python run.py

# Or specify a custom port
python run.py 3000
```

The script will:
- ✅ Start a local HTTP server
- ✅ Automatically open your browser to the application
- ✅ Show you helpful information and tips

Alternatively, use Python's built-in server:

```bash
python -m http.server 8000
# Then manually open: http://localhost:8000/index.html
```

## 📖 How to Use

### 1. Upload TIFF File

- **Drag & Drop**: Drag your `.tiff` or `.tif` file onto the upload area
- **Browse**: Click "Choose TIFF File" button to select a file
- Supported formats: `.tiff`, `.tif` (16-bit grayscale recommended)

### 2. Adjust Parameters

The application provides several adjustable parameters:

| Parameter | Description | Default | Range | Unit |
|-----------|-------------|---------|-------|------|
| **Threshold** | Pixel intensity threshold for object separation | 24000 | 1000-65535 | pixel value |
| **Min Area** | Minimum contour area to consider | 2000 | 100-50000 | px² (square pixels) |
| **Max Area** | Maximum contour area to consider | 100000 | 10000-500000 | px² (square pixels) |
| **Min Circularity** | How circular objects must be (0=any, 1=perfect circle) | 0.6 | 0.1-1.0 | ratio (0-1) |
| **Min Solidity** | Object density filter (removes hollow objects) | 0.7 | 0.1-1.0 | ratio (0-1) |
| **Expected Count** | Number of objects expected (for diagonal detection) | 4 | 1-20 | count |
| **Grid Columns** | Number of columns in grid layout | 4 | 2-10 | count |

### 3. Process Image

Click the **"🚀 Process Image"** button to start analysis. The processing includes:

1. ✅ Image loading and validation
2. ✅ Binary thresholding
3. ✅ Noise reduction with morphological operations
4. ✅ Contour detection
5. ✅ Shape filtering (circularity, solidity, aspect ratio)
6. ✅ Duplicate removal
7. ✅ Grid position calculation
8. ✅ Pixel value analysis and classification

### 4. View Results

The application displays:

- **Detection Result**: Image with detected circles highlighted
- **Threshold Mask**: Binary mask showing detected regions
- **Statistics Table**: Detailed data for each detected circle
  - Position coordinates
  - Radius
  - Mean pixel value
  - Shape metrics (circularity, solidity)
  - Classification (Hitam/Abu-abu)
- **Grid Analysis**: Full 4x4 grid visualization with predicted positions

## 🎨 Understanding the Output

### Color Coding

- 🔴 **Red Circles**: Dark objects (pixel value < 6000) - classified as "Hitam"
- 🟢 **Green Circles**: Light objects (pixel value ≥ 6000) - classified as "Abu-abu"
- 🟡 **Yellow Circles**: Predicted grid positions in grid analysis
- 🔵 **Blue Numbers**: Circle identification numbers

### Metrics Explained

- **Circularity**: How close to a perfect circle (1.0 = perfect circle)
- **Solidity**: Ratio of contour area to convex hull area (detects hollow objects)
- **Aspect Ratio**: Width/height ratio (detects oval shapes)
- **Mean Value**: Average pixel intensity within the circle

## 🔧 Troubleshooting

### Issue: PyScript not loading
**Solution**: 
- Check your internet connection
- Try refreshing the page
- Clear browser cache
- Ensure JavaScript is enabled

### Issue: Image not processing
**Solution**:
- Verify the file is a valid TIFF format
- Check file size (very large files may take time)
- Try adjusting threshold parameters
- Check browser console for error messages (F12)

### Issue: No circles detected
**Solution**:
- Adjust threshold value (try different values between 10000-30000)
- Lower minimum circularity and solidity values
- Adjust min/max area parameters
- Verify image quality and contrast

### Issue: Too many/few circles detected
**Solution**:
- Fine-tune the threshold value
- Adjust circularity and solidity for stricter/looser filtering
- Change expected count to match your image
- Modify min/max area constraints

## 📁 File Structure

```
imageAnalysMiu/
├── index.html           # Main web application
├── processor.py         # PyScript processing module
├── circle-detection.ipynb  # Original Jupyter notebook
└── README.md           # This file
```

## 🌟 Features in Detail

### Diagonal Pattern Detection

The application can detect circles along a diagonal and extrapolate to find all positions in a 4x4 grid:

1. Detects circles along the main diagonal
2. Calculates spacing between diagonal elements
3. Generates all 16 grid positions based on spacing
4. Visualizes complete grid with predicted centers

### Adaptive Filtering

Multiple filtering stages ensure accurate detection:

- **Area filtering**: Removes too small/large objects
- **Circularity filtering**: Ensures round shapes
- **Solidity filtering**: Removes hollow or irregular objects
- **Aspect ratio filtering**: Removes elongated shapes
- **Duplicate removal**: Eliminates overlapping detections

## 🔐 Privacy & Security

- ✅ **100% Client-Side**: All processing happens in your browser
- ✅ **No Data Upload**: Your images never leave your computer
- ✅ **No Backend**: No server-side code or database
- ✅ **Open Source**: All code is visible and auditable

## 🤝 Contributing

This project is part of the imageAnalysMiu analysis toolkit. Feel free to:

- Report issues
- Suggest improvements
- Submit pull requests
- Share your use cases

## 📝 License

This project is provided as-is for educational and research purposes.

## 🎓 Use Cases

Perfect for:

- 🔬 Laboratory sample analysis
- 🏭 Quality control inspection
- 🔍 Pattern recognition research
- 📊 Automated measurement systems
- 🎯 Object counting and classification

## 💡 Tips for Best Results

1. **Image Quality**: Use high-quality 16-bit TIFF images
2. **Contrast**: Ensure good contrast between objects and background
3. **Lighting**: Uniform lighting produces better results
4. **Parameter Tuning**: Start with defaults, then adjust incrementally
5. **Grid Detection**: Ensure at least 2-4 diagonal circles are detected first

## 🆘 Support

For issues or questions:

1. Check the Troubleshooting section above
2. Review the original `circle-detection.ipynb` notebook
3. Open browser console (F12) to see detailed error messages
4. Verify PyScript initialization completed successfully

## 🔄 Version History

### Version 1.0 (Current)
- Initial web-based implementation
- PyScript integration
- Full circle detection pipeline
- Grid analysis and visualization
- Responsive UI with drag-and-drop
- Real-time parameter adjustment

---

**Made with ❤️ using PyScript - Python for the Web!**

*No installation. No backend. Just upload and analyze!* 🚀
