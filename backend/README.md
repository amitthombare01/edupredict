# Student Performance Analysis (Backend)

This folder contains the **backend** code for student performance analysis.
The backend exposes a simple API that serves student performance data.

## Structure

- `data/` - raw and processed datasets (CSV, etc.)
- `notebooks/` - exploratory analysis and visualization notebooks
- `src/` - core analysis code and modules
- `tests/` - unit tests

## Getting Started

1. Change into the backend folder:

```powershell
cd backend
```

2. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the API (default port 8000):

```powershell
python app.py
```

To use a different port (e.g., 8010) from PowerShell, set `PORT` before running:

```powershell
$env:PORT = 8010
python app.py
```

This keeps the backend accessible at `http://127.0.0.1:<PORT>` and is the port the frontend dev server should proxy to (see the root README for how to align both sides).

### (Optional) Run via `npm run dev`

```
cd backend
npm install
npm run dev
```

The npm script simply launches the activated virtual environment (`.venv\Scripts\python.exe app.py`), so it behaves the same as the Python command above and still respects `PORT` if you need to override it.

## API Endpoints

- `GET /api/students` — returns student performance records.
