# Quick Deployment Instructions

## Your Repository
✅ Created: https://github.com/Ajitesh-007/injury-detection-app

---

## Deploy Now (2 Simple Steps)

### Step 1: Deploy Backend to Render
1. Open: https://dashboard.render.com/
2. Click "New +" → "Web Service"
3. Connect repository: `Ajitesh-007/injury-detection-app`
4. Settings:
   - **Name**: `injury-detection-backend`
   - **Root Directory**: `backend`
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: Free
5. Click "Create Web Service"
6. **Copy the URL** (e.g., `https://injury-detection-backend.onrender.com`)

### Step 2: Deploy Frontend to Vercel  
1. Open: https://vercel.com/new
2. Import `Ajitesh-007/injury-detection-app`
3. Settings:
   - **Framework**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Add Environment Variable:
   - **Name**: `VITE_API_URL`
   - **Value**: (paste your Render URL from Step 1)
5. Click "Deploy"
6. **Your app is live!** Copy the Vercel URL

---

## ✅ Done!

Your app will be accessible at the Vercel URL from any computer in the world!
