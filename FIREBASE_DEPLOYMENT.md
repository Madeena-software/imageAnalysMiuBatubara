# 🔥 Firebase Hosting Deployment Guide

This guide will help you deploy the Image Analysis Tool to Firebase Hosting.

## ✅ What's Included

The `public/` folder contains everything needed for Firebase deployment:
- `index.html` - Main application
- `processor.py` - Python processing module (loaded by PyScript)

## 📋 Prerequisites

1. **Node.js and npm** installed on your system
   - Download from: https://nodejs.org/
   - Check installation: `node --version` and `npm --version`

2. **Firebase CLI** installed globally
   ```bash
   npm install -g firebase-tools
   ```

3. **Firebase Account**
   - Create a free account at https://firebase.google.com/
   - Create a new project in the Firebase Console

## 🚀 Deployment Steps

### Step 1: Login to Firebase

```bash
firebase login
```

This will open a browser window for authentication.

### Step 2: Initialize Firebase Project (First Time Only)

If this is your first deployment, run:

```bash
firebase init
```

Select these options:
- **Which Firebase features?** → Choose "Hosting"
- **Select a project** → Choose your Firebase project (or create new one)
- **What folder to use?** → `public` (already configured)
- **Configure as single-page app?** → No
- **Set up automatic builds?** → No
- **Overwrite index.html?** → No

**OR** simply update the project ID in `.firebaserc`:

Edit `.firebaserc` and replace `your-project-id` with your actual Firebase project ID:
```json
{
  "projects": {
    "default": "your-actual-project-id"
  }
}
```

### Step 3: Deploy to Firebase

```bash
firebase deploy
```

Wait for deployment to complete. You'll see a hosting URL like:
```
✔  Deploy complete!

Hosting URL: https://your-project-id.web.app
```

### Step 4: Test Your Deployment

Open the provided URL in your browser. The first load will take **30-60 seconds** as PyScript downloads Python packages (OpenCV, NumPy, Pillow, Matplotlib).

## ⚙️ Configuration Details

### firebase.json

The configuration includes:
- **CORS Headers**: Required for PyScript to work properly
- **Caching**: Optimized cache settings for better performance
- **Security Headers**: Cross-Origin policies for isolated execution

### Performance Notes

⏱️ **First Load**: 30-60 seconds (PyScript downloads ~50MB of Python packages)
⚡ **Subsequent Loads**: Much faster due to browser caching
🔄 **Processing**: Happens client-side in the browser (no server needed!)

## 🛠️ Updating Your Deployment

To update after making changes:

1. Make changes to files in the main directory
2. Copy updated files to `public/` if needed:
   ```bash
   Copy-Item ".\index.html" -Destination ".\public\index.html" -Force
   Copy-Item ".\processor.py" -Destination ".\public\processor.py" -Force
   ```
3. Deploy again:
   ```bash
   firebase deploy
   ```

## 📊 Monitoring

View deployment status and analytics:
```bash
firebase open hosting
```

## 🔍 Troubleshooting

### Issue: CORS Errors
- Make sure the headers in `firebase.json` are properly configured
- Clear browser cache and try again

### Issue: PyScript Not Loading
- Check browser console for errors
- Verify internet connection (PyScript needs to download packages)
- Try in incognito/private mode to rule out extension conflicts

### Issue: Slow Initial Load
- This is normal! PyScript downloads ~50MB on first load
- Consider adding a loading indicator (already included in the app)
- Packages are cached after first load

### Issue: File Upload Errors
- Check that you're uploading valid TIFF files
- Try with smaller images first to verify functionality
- Check browser console for detailed error messages

## 💰 Cost

Firebase Hosting free tier includes:
- **Storage**: 10 GB
- **Transfer**: 360 MB/day
- **Custom domain**: Supported

This should be more than enough for this application since all processing happens client-side!

## 🔒 Security Considerations

- All processing happens in the user's browser
- Images never leave the client (privacy-friendly!)
- No server-side storage or processing
- Cross-Origin policies prevent malicious access

## 📱 Browser Compatibility

Works on modern browsers:
- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 15.4+
- ❌ Internet Explorer (not supported)

## 🆘 Getting Help

- Firebase Documentation: https://firebase.google.com/docs/hosting
- PyScript Documentation: https://docs.pyscript.net/
- Firebase Support: https://firebase.google.com/support

---

**🎉 You're all set!** Your image analysis tool is now deployed to Firebase Hosting and accessible worldwide.
