# Student Performance Analysis

This repository contains both a **backend** (data analysis API) and a **frontend** (UI) for exploring student performance.

## Structure

- `backend/` - Python backend + analysis code
- `frontend/` - UI for interacting with analysis results

## Getting Started

### Backend

1. Open PowerShell in this repo.
2. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

4. Run the backend (FastAPI):

```powershell
python -m backend.app
```

If you ever need to start the backend on a different port (e.g., 8010 because 8000 is taken), set `PORT` before launching:

```powershell
$env:PORT=8010
python -m backend.app
```

Make sure the frontend dev server mirrors that port via `API_PORT` (see below).

### Deploy On Render

1. Push this repository to GitHub.
2. In Render, create a new `Blueprint` or `Web Service` from the repo.
3. If you use the included `render.yaml`, Render will pick these automatically:

```yaml
buildCommand: pip install -r backend/requirements.txt
startCommand: cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT
```

4. Add these environment variables in Render:

```text
MONGODB_URI=<your Render MongoDB connection string>
MONGODB_DB=student_ai
```

5. Deploy, then open:

```text
https://<your-render-service>.onrender.com
```

Notes:
- The app now reads MongoDB from `MONGODB_URI` instead of assuming localhost.
- The frontend is served by the FastAPI app, so you only need one Render web service for this repo.
- Health check path: `/api/health`

### Frontend

#### Option 1: Simple (no build)

Open `frontend/index.html` directly in a browser. It will fetch data from the backend.

#### Option 2: `npm run dev` (recommended for a local dev server)

From the `frontend/` folder:

```powershell
npm install
$env:API_PORT=8010
npm run dev
```

Then open the URL shown in the console (usually `http://localhost:5173`). The dev server proxies `/api` to `http://127.0.0.1:${API_PORT || PORT || 8000}` and will respect any backend port overrides.
