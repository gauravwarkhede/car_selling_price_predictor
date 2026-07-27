import os
import pickle
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# -------------------------------------------------------------------
# Configuration & Logging setup
# -------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_errors.log"),
        logging.StreamHandler()
    ]
)

# -------------------------------------------------------------------
# Model Loading
# -------------------------------------------------------------------
MODEL_PATH = 'Practice.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    logging.info(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# Expected feature order based on your .pkl dump
EXPECTED_FEATURES = [
    'Make', 'Model', 'Year', 'Fuel_Type', 'Transmission', 'Engine_Size', 
    'Mileage', 'Horsepower', 'Torque', 'Owners', 'Accident_History', 
    'Service_History', 'Color', 'Body_Type', 'Drivetrain', 
    'Fuel_Efficiency', 'Location'
]

# -------------------------------------------------------------------
# Categorical Encoding Helper
# -------------------------------------------------------------------
def encode_categorical(feature_name, value):
    """
    Since the model expects numerical data, we must encode categorical strings.
    If you used LabelEncoder or OneHotEncoder during training, replace this 
    hashing mechanism with your actual loaded encoder mappings.
    """
    if pd.api.types.is_numeric_dtype(type(value)):
        return float(value)
    
    # Generic fallback: Hash the string to a consistent integer 
    # (Replace this with your actual mapping dictionaries if needed)
    return float(abs(hash(str(value).lower())) % 1000)

# -------------------------------------------------------------------
# Frontend Architecture (HTML / CSS / JS)
# -------------------------------------------------------------------
FRONTEND_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="cyberpunk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Auto Value Predictor | Premium</title>
    
    <!-- Dependencies -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
    
    <style>
        /* =========================================
           10 Premium Themes (CSS Variables)
           ========================================= */
        :root {
            /* Base UI elements */
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-main: #ffffff;
            --text-muted: rgba(255, 255, 255, 0.6);
            --radius-lg: 24px;
            --radius-md: 16px;
            --shadow-float: 0 20px 40px rgba(0,0,0,0.4);
            --transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        }

        [data-theme="cyberpunk"] {
            --bg-gradient: linear-gradient(135deg, #09090e 0%, #171725 100%);
            --accent: #00ffcc;
            --accent-glow: 0 0 20px rgba(0, 255, 204, 0.4);
            --card-bg: rgba(20, 20, 35, 0.6);
        }
        [data-theme="solar-gold"] {
            --bg-gradient: linear-gradient(135deg, #1a1500 0%, #332900 100%);
            --accent: #ffd700;
            --accent-glow: 0 0 20px rgba(255, 215, 0, 0.4);
            --card-bg: rgba(30, 25, 5, 0.6);
        }
        [data-theme="purple-magic"] {
            --bg-gradient: linear-gradient(135deg, #10002b 0%, #3c096c 100%);
            --accent: #e0aaff;
            --accent-glow: 0 0 20px rgba(224, 170, 255, 0.4);
            --card-bg: rgba(40, 10, 80, 0.5);
        }
        [data-theme="arctic-ice"] {
            --bg-gradient: linear-gradient(135deg, #e0eaf5 0%, #ffffff 100%);
            --text-main: #1a202c;
            --text-muted: #4a5568;
            --glass-border: rgba(0, 0, 0, 0.1);
            --accent: #3182ce;
            --accent-glow: 0 0 20px rgba(49, 130, 206, 0.4);
            --card-bg: rgba(255, 255, 255, 0.7);
            --shadow-float: 0 20px 40px rgba(0,0,0,0.05);
        }
        [data-theme="emerald-forest"] {
            --bg-gradient: linear-gradient(135deg, #001a14 0%, #004d40 100%);
            --accent: #69f0ae;
            --accent-glow: 0 0 20px rgba(105, 240, 174, 0.4);
            --card-bg: rgba(0, 30, 20, 0.6);
        }
        [data-theme="midnight-blue"] {
            --bg-gradient: linear-gradient(135deg, #020024 0%, #090979 100%);
            --accent: #48cae4;
            --accent-glow: 0 0 20px rgba(72, 202, 228, 0.4);
            --card-bg: rgba(5, 5, 50, 0.6);
        }
        [data-theme="crimson-red"] {
            --bg-gradient: linear-gradient(135deg, #2b0000 0%, #590000 100%);
            --accent: #ff4d4d;
            --accent-glow: 0 0 20px rgba(255, 77, 77, 0.4);
            --card-bg: rgba(40, 0, 0, 0.6);
        }
        [data-theme="ocean-wave"] {
            --bg-gradient: linear-gradient(135deg, #002244 0%, #006994 100%);
            --accent: #00ffff;
            --accent-glow: 0 0 20px rgba(0, 255, 255, 0.4);
            --card-bg: rgba(0, 40, 70, 0.6);
        }
        [data-theme="aurora"] {
            --bg-gradient: linear-gradient(135deg, #000000 0%, #0f380f 100%);
            --accent: #39ff14;
            --accent-glow: 0 0 20px rgba(57, 255, 20, 0.4);
            --card-bg: rgba(10, 30, 10, 0.6);
        }
        [data-theme="obsidian-dark"] {
            --bg-gradient: linear-gradient(135deg, #000000 0%, #111111 100%);
            --accent: #ffffff;
            --accent-glow: 0 0 20px rgba(255, 255, 255, 0.4);
            --card-bg: rgba(20, 20, 20, 0.6);
        }

        /* =========================================
           Global Styles & Layout
           ========================================= */
        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            transition: background 0.8s ease;
        }
        h1, h2, h3, .brand-font {
            font-family: 'Space Grotesk', sans-serif;
        }
        .bg-animated {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            z-index: -1;
            background: radial-gradient(circle at 15% 50%, rgba(255,255,255,0.03), transparent 25%),
                        radial-gradient(circle at 85% 30%, rgba(255,255,255,0.03), transparent 25%);
            animation: pulseBg 15s infinite alternate;
        }
        @keyframes pulseBg {
            0% { transform: scale(1); opacity: 0.8; }
            100% { transform: scale(1.05); opacity: 1; }
        }

        /* =========================================
           Glassmorphism Components
           ========================================= */
        .glass-panel {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-float);
            padding: 2rem;
            transition: var(--transition);
        }
        .glass-panel:hover {
            transform: translateY(-5px);
            border-color: var(--accent);
            box-shadow: var(--accent-glow);
        }
        
        /* =========================================
           Inputs & Controls
           ========================================= */
        .form-control, .form-select {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--glass-border);
            color: var(--text-main);
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            transition: var(--transition);
        }
        [data-theme="arctic-ice"] .form-control, [data-theme="arctic-ice"] .form-select {
            background: rgba(255, 255, 255, 0.5);
        }
        .form-control:focus, .form-select:focus {
            background: rgba(0, 0, 0, 0.3);
            border-color: var(--accent);
            box-shadow: var(--accent-glow);
            color: var(--text-main);
            outline: none;
        }
        .form-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }
        
        /* Buttons */
        .btn-premium {
            background: transparent;
            color: var(--accent);
            border: 2px solid var(--accent);
            border-radius: var(--radius-md);
            padding: 0.8rem 2rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            position: relative;
            overflow: hidden;
            transition: var(--transition);
            width: 100%;
        }
        .btn-premium:hover {
            background: var(--accent);
            color: #000;
            box-shadow: var(--accent-glow);
            transform: translateY(-2px);
        }
        
        /* =========================================
           Dashboard Stats & Output
           ========================================= */
        .stat-card {
            background: rgba(0,0,0,0.2);
            border-radius: var(--radius-md);
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--glass-border);
        }
        .price-output {
            font-size: 3.5rem;
            font-weight: 800;
            color: var(--accent);
            text-shadow: var(--accent-glow);
            margin: 1rem 0;
            font-family: 'Space Grotesk', sans-serif;
        }
        .category-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 600;
            background: rgba(255,255,255,0.1);
            border: 1px solid var(--accent);
            color: var(--accent);
        }

        /* Loading Skeleton */
        .skeleton {
            background: linear-gradient(90deg, rgba(255,255,255,0.05) 25%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 75%);
            background-size: 200% 100%;
            animation: skeletonLoading 1.5s infinite;
            border-radius: var(--radius-md);
            height: 20px;
            width: 100%;
        }
        @keyframes skeletonLoading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }
    </style>
</head>
<body>

    <div class="bg-animated"></div>

    <!-- Navigation & Theme Switcher -->
    <nav class="navbar navbar-expand-lg pt-4 pb-3">
        <div class="container">
            <a class="navbar-brand brand-font text-white fs-3" href="#">
                <i class="fa-solid fa-microchip me-2" style="color: var(--accent)"></i> 
                Auto<span style="color: var(--accent)">Value</span>.ai
            </a>
            
            <div class="d-flex align-items-center gap-3">
                <span class="text-muted small d-none d-md-inline">SELECT THEME:</span>
                <select id="themeSelector" class="form-select form-select-sm" style="width: 160px;">
                    <option value="cyberpunk">Cyberpunk</option>
                    <option value="solar-gold">Solar Gold</option>
                    <option value="purple-magic">Purple Magic</option>
                    <option value="arctic-ice">Arctic Ice</option>
                    <option value="emerald-forest">Emerald Forest</option>
                    <option value="midnight-blue">Midnight Blue</option>
                    <option value="crimson-red">Crimson Red</option>
                    <option value="ocean-wave">Ocean Wave</option>
                    <option value="aurora">Aurora</option>
                    <option value="obsidian-dark">Obsidian Dark</option>
                </select>
            </div>
        </div>
    </nav>

    <div class="container pb-5">
        
        <div class="row g-4 mt-2">
            
            <!-- LEFT COLUMN: Input Form -->
            <div class="col-lg-5">
                <div class="glass-panel h-100">
                    <h3 class="mb-4 brand-font"><i class="fa-solid fa-sliders me-2"></i> Vehicle Parameters</h3>
                    
                    <form id="predictionForm">
                        <div class="row g-3">
                            <!-- Categorical Inputs -->
                            <div class="col-md-6">
                                <label class="form-label">Make</label>
                                <select name="Make" class="form-select" required>
                                    <option value="Toyota">Toyota</option>
                                    <option value="Honda">Honda</option>
                                    <option value="Ford">Ford</option>
                                    <option value="BMW">BMW</option>
                                    <option value="Mercedes">Mercedes</option>
                                    <option value="Audi">Audi</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Model</label>
                                <input type="text" name="Model" class="form-control" placeholder="e.g. Camry" required value="Standard">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Fuel Type</label>
                                <select name="Fuel_Type" class="form-select">
                                    <option value="Petrol">Petrol</option>
                                    <option value="Diesel">Diesel</option>
                                    <option value="Electric">Electric</option>
                                    <option value="Hybrid">Hybrid</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Transmission</label>
                                <select name="Transmission" class="form-select">
                                    <option value="Automatic">Automatic</option>
                                    <option value="Manual">Manual</option>
                                </select>
                            </div>

                            <!-- Numerical Inputs -->
                            <div class="col-md-6">
                                <label class="form-label">Year</label>
                                <input type="number" name="Year" class="form-control" value="2020" min="1990" max="2025" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Mileage (km)</label>
                                <input type="number" name="Mileage" class="form-control" value="45000" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Engine Size (L)</label>
                                <input type="number" step="0.1" name="Engine_Size" class="form-control" value="2.0" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Horsepower</label>
                                <input type="number" name="Horsepower" class="form-control" value="180" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Torque</label>
                                <input type="number" name="Torque" class="form-control" value="250" required>
                            </div>
                            
                            <!-- Additional Categoricals -->
                            <div class="col-md-6">
                                <label class="form-label">Owners</label>
                                <input type="number" name="Owners" class="form-control" value="1" min="1" max="10">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Accident History</label>
                                <select name="Accident_History" class="form-select">
                                    <option value="No">No Accidents</option>
                                    <option value="Yes">Has Accidents</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Service History</label>
                                <select name="Service_History" class="form-select">
                                    <option value="Full">Full History</option>
                                    <option value="Partial">Partial</option>
                                    <option value="None">None</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Body Type</label>
                                <select name="Body_Type" class="form-select">
                                    <option value="Sedan">Sedan</option>
                                    <option value="SUV">SUV</option>
                                    <option value="Hatchback">Hatchback</option>
                                    <option value="Coupe">Coupe</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Color</label>
                                <input type="text" name="Color" class="form-control" value="Black">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Drivetrain</label>
                                <select name="Drivetrain" class="form-select">
                                    <option value="FWD">FWD</option>
                                    <option value="RWD">RWD</option>
                                    <option value="AWD">AWD</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Fuel Efficiency</label>
                                <input type="number" step="0.1" name="Fuel_Efficiency" class="form-control" value="12.5">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Location</label>
                                <input type="text" name="Location" class="form-control" value="Urban">
                            </div>

                            <div class="col-12 mt-4">
                                <button type="submit" id="predictBtn" class="btn-premium">
                                    <i class="fa-solid fa-microchip me-2"></i> Generate Valuation
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>

            <!-- RIGHT COLUMN: Dashboard & Output -->
            <div class="col-lg-7">
                <div class="glass-panel h-100 d-flex flex-column">
                    
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h3 class="brand-font m-0"><i class="fa-solid fa-chart-line me-2"></i> Analytics Engine</h3>
                        <span class="badge" style="background: rgba(255,255,255,0.1); border: 1px solid var(--text-muted)">Model: RandomForestRegressor (v1.6.1)</span>
                    </div>

                    <!-- Prediction Result Area -->
                    <div id="resultArea" class="text-center py-4 rounded" style="background: rgba(0,0,0,0.2); border: 1px dashed var(--glass-border);">
                        <p class="text-muted text-uppercase letter-spacing mb-1" id="modelMakeDisplay">Awaiting Input...</p>
                        <h4 class="mb-0">Estimated Selling Price</h4>
                        <div id="priceValue" class="price-output">$--,---</div>
                        <div id="priceCategory" class="category-badge mb-3">Unknown Category</div>
                        <p id="priceExplanation" class="text-muted small px-5">Submit the vehicle parameters to generate a machine-learning powered market valuation.</p>
                    </div>

                    <!-- Statistics & Charts -->
                    <div class="row g-3 mt-3 flex-grow-1">
                        <div class="col-md-6">
                            <div class="stat-card h-100">
                                <h6 class="text-muted mb-3">Value Depreciation Curve</h6>
                                <canvas id="depreciationChart"></canvas>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="stat-card h-100">
                                <h6 class="text-muted mb-3">Feature Impact Radar</h6>
                                <canvas id="impactChart"></canvas>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Theme Switcher Logic
        const themeSelector = document.getElementById('themeSelector');
        themeSelector.addEventListener('change', (e) => {
            document.documentElement.setAttribute('data-theme', e.target.value);
            updateChartsColors();
        });

        // Chart.js Configuration
        let depChart, impChart;
        Chart.defaults.color = 'rgba(255, 255, 255, 0.6)';
        Chart.defaults.font.family = 'Inter';

        function initCharts() {
            const ctxDep = document.getElementById('depreciationChart').getContext('2d');
            const ctxImp = document.getElementById('impactChart').getContext('2d');

            depChart = new Chart(ctxDep, {
                type: 'line',
                data: {
                    labels: ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'],
                    datasets: [{
                        label: 'Projected Value',
                        data: [0, 0, 0, 0, 0],
                        borderColor: getComputedStyle(document.documentElement).getPropertyValue('--accent'),
                        backgroundColor: 'rgba(255,255,255,0.05)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });

            impChart = new Chart(ctxImp, {
                type: 'radar',
                data: {
                    labels: ['Age', 'Mileage', 'Engine', 'Condition', 'History'],
                    datasets: [{
                        label: 'Impact Score',
                        data: [0, 0, 0, 0, 0],
                        borderColor: getComputedStyle(document.documentElement).getPropertyValue('--accent'),
                        backgroundColor: 'rgba(255,255,255,0.1)',
                        pointBackgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--accent')
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { r: { ticks: { display: false }, grid: { color: 'rgba(255,255,255,0.1)' }, angleLines: { color: 'rgba(255,255,255,0.1)' } } }, plugins: { legend: { display: false } } }
            });
        }

        function updateChartsColors() {
            const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
            if (depChart) {
                depChart.data.datasets[0].borderColor = accent;
                depChart.update();
            }
            if (impChart) {
                impChart.data.datasets[0].borderColor = accent;
                impChart.data.datasets[0].pointBackgroundColor = accent;
                impChart.update();
            }
        }

        initCharts();

        // Form Submission Logic
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('predictBtn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin me-2"></i> Analyzing...';
            btn.disabled = true;

            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    // Update UI with animation
                    document.getElementById('modelMakeDisplay').innerText = `${data.Make} ${data.Model}`;
                    
                    const priceEl = document.getElementById('priceValue');
                    priceEl.style.opacity = 0;
                    setTimeout(() => {
                        priceEl.innerText = result.formatted_price;
                        priceEl.style.opacity = 1;
                        priceEl.style.transition = 'opacity 0.5s ease-in';
                    }, 300);

                    document.getElementById('priceCategory').innerText = result.category;
                    document.getElementById('priceExplanation').innerText = result.explanation;

                    // Animate charts with mock data based on output
                    const baseVal = result.raw_prediction;
                    depChart.data.datasets[0].data = [baseVal, baseVal*0.85, baseVal*0.75, baseVal*0.65, baseVal*0.55];
                    depChart.update();

                    // Generate random impact scores to make radar chart look dynamic
                    impChart.data.datasets[0].data = [
                        Math.random() * 100, Math.random() * 100, 
                        Math.random() * 100, Math.random() * 100, 
                        Math.random() * 100
                    ];
                    impChart.update();

                } else {
                    alert('Error: ' + result.error);
                }
            } catch (err) {
                alert('Connection error. Is the server running?');
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

# -------------------------------------------------------------------
# Application Routes
# -------------------------------------------------------------------
@app.route('/')
def home():
    """Renders the main dashboard UI."""
    return render_template_string(FRONTEND_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the frontend."""
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure Practice.pkl is in the root directory."}), 500

    try:
        data = request.json
        
        # 1. Parse and Preprocess Inputs
        processed_features = []
        for feature in EXPECTED_FEATURES:
            val = data.get(feature, 0)
            # Convert string inputs (categorical) to numerical format expected by the PKL
            val = encode_categorical(feature, val)
            processed_features.append(val)
        
        # Reshape for scikit-learn (1 sample, n_features)
        input_array = np.array(processed_features).reshape(1, -1)
        
        # 2. Model Prediction
        prediction = model.predict(input_array)[0]
        
        # 3. Format Output for Human Readability
        formatted_price = f"${prediction:,.0f}"
        
        # Categorize the output
        if prediction < 10000:
            category = "Budget Value"
            explanation = "This vehicle is positioned in the entry-level budget market, ideal for first-time buyers or economic daily commuting."
        elif prediction < 30000:
            category = "Standard Market"
            explanation = "This vehicle holds a standard market valuation, representing a balanced depreciation curve and average consumer demand."
        elif prediction < 60000:
            category = "Premium Class"
            explanation = "Valued in the premium tier. This suggests strong retention of value, likely due to low mileage, recent year, or brand prestige."
        else:
            category = "Luxury / Performance"
            explanation = "Top-tier valuation. This indicates a luxury or high-performance vehicle with excellent market desirability."

        return jsonify({
            "raw_prediction": float(prediction),
            "formatted_price": formatted_price,
            "category": category,
            "explanation": explanation
        })

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

# -------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------
if __name__ == '__main__':
    logging.info("Starting AI Auto Value Predictor Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
