// app.js - Complete DataYab web interface

const API_BASE = '/api';
let stackingDialogOpen = false;

// ================================
// Navigation
// ================================
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function() {
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        this.classList.add('active');
        
        const section = this.dataset.section;
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById(section).classList.add('active');
        
        // Load data for section
        if (section === 'explore') loadPreview();
        if (section === 'predict') loadPredictForm();
        if (section === 'evaluate') loadModels();
        if (section === 'feature') loadFeatureStatus();
    });
});

// ================================
// Upload
// ================================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

if (uploadArea) {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFile(files[0]);
    });
}

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) uploadFile(e.target.files[0]);
    });
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        const resultDiv = document.getElementById('uploadResult');
        
        if (result.success) {
            resultDiv.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i> ${result.message}<br>
                    <strong>Shape:</strong> ${result.shape}<br>
                    <strong>Columns:</strong> ${result.columns.slice(0, 5).join(', ')}${result.columns.length > 5 ? '...' : ''}
                </div>
            `;
            updateStatus('Data loaded ✓', 'success');
            loadColumns();
        } else {
            resultDiv.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-circle"></i> ${result.message}</div>`;
        }
    } catch (e) {
        alert('Upload error: ' + e.message);
    }
}

// ================================
// Preview
// ================================
async function loadPreview() {
    try {
        const response = await fetch(`${API_BASE}/preview?rows=10`);
        const result = await response.json();
        
        if (result.success) {
            const table = document.getElementById('previewTable');
            table.querySelector('thead').innerHTML = `
                <tr>
                    ${result.columns.map(col => `<th>${col}</th>`).join('')}
                </tr>
            `;
            table.querySelector('tbody').innerHTML = result.data.map(row => `
                <tr>
                    ${row.map(cell => `<td>${cell}</td>`).join('')}
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error('Preview error:', e);
    }
}

// ================================
// Columns
// ================================
async function loadColumns() {
    try {
        const response = await fetch(`${API_BASE}/columns`);
        const result = await response.json();
        
        if (result.success) {
            const cols = result.columns;
            
            // Update all column dropdowns
            ['targetColumn', 'chartColumn'].forEach(id => {
                const select = document.getElementById(id);
                if (select) {
                    select.innerHTML = cols.map(col => `<option value="${col}">${col}</option>`).join('');
                }
            });
        }
    } catch (e) {
        console.error('Columns error:', e);
    }
}

// ================================
// Clean
// ================================
async function cleanData(operation) {
    const target = document.getElementById('targetColumn')?.value;
    
    try {
        const response = await fetch(`${API_BASE}/clean?operation=${operation}&target=${target}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        const resultDiv = document.getElementById('cleanResult');
        
        if (result.success) {
            if (result.steps) {
                resultDiv.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle"></i> Cleaning complete!<br>
                        <ul class="mt-2">
                            ${result.steps.map(step => `<li>${step}</li>`).join('')}
                        </ul>
                    </div>
                `;
            } else if (result.encoded) {
                resultDiv.innerHTML = `<div class="alert alert-success"><i class="fas fa-check-circle"></i> Encoded: ${result.encoded.join(', ')}</div>`;
            } else if (result.removed !== undefined) {
                resultDiv.innerHTML = `<div class="alert alert-success"><i class="fas fa-check-circle"></i> Removed: ${result.removed}</div>`;
            } else {
                resultDiv.innerHTML = `<div class="alert alert-success"><i class="fas fa-check-circle"></i> ${result.message || 'Operation complete'}</div>`;
            }
            updateStatus('Data cleaned ✓', 'success');
            loadColumns();
        } else {
            resultDiv.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-circle"></i> ${result.message || result.detail}</div>`;
        }
    } catch (e) {
        alert('Clean error: ' + e.message);
    }
}

// ================================
// Feature Selection
// ================================
async function getRFImportance() {
    try {
        const response = await fetch(`${API_BASE}/feature-importance-rf`);
        const result = await response.json();
        
        if (result.success && result.importance) {
            let html = '<div class="alert alert-info"><strong>🌲 Random Forest Feature Importance</strong><ul>';
            const maxVal = Math.max(...Object.values(result.importance));
            for (const [feature, importance] of Object.entries(result.importance).slice(0, 15)) {
                const barWidth = (importance * 100) / maxVal;
                html += `<li>${feature}: ${importance.toFixed(4)} 
                    <div class="progress mt-1" style="height: 8px;">
                        <div class="progress-bar" style="width: ${barWidth}%;"></div>
                    </div>
                </li>`;
            }
            html += '</ul><button class="btn btn-sm btn-primary" onclick="keepTopFeatures()">Keep Top Features</button></div>';
            document.getElementById('featureImportanceResult').innerHTML = html;
        } else if (result.detail) {
            document.getElementById('featureImportanceResult').innerHTML = `<div class="alert alert-warning">${result.detail}</div>`;
        }
    } catch (e) {
        document.getElementById('featureImportanceResult').innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

async function getMutualInfo() {
    try {
        const response = await fetch(`${API_BASE}/mutual-info`);
        const result = await response.json();
        
        if (result.success && result.importance) {
            let html = '<div class="alert alert-info"><strong>📊 Mutual Information Scores</strong><ul>';
            const maxVal = Math.max(...Object.values(result.importance));
            for (const [feature, score] of Object.entries(result.importance).slice(0, 15)) {
                const barWidth = (score / maxVal) * 100;
                html += `<li>${feature}: ${score.toFixed(6)} 
                    <div class="progress mt-1" style="height: 8px;">
                        <div class="progress-bar" style="width: ${barWidth}%;"></div>
                    </div>
                </li>`;
            }
            html += '</ul><button class="btn btn-sm btn-primary" onclick="keepTopFeatures()">Keep Top Features</button></div>';
            document.getElementById('featureImportanceResult').innerHTML = html;
        } else if (result.detail) {
            document.getElementById('featureImportanceResult').innerHTML = `<div class="alert alert-warning">${result.detail}</div>`;
        }
    } catch (e) {
        document.getElementById('featureImportanceResult').innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

async function applyManualSelection() {
    const features = document.getElementById('manualFeatures').value;
    if (!features) {
        alert('Enter feature names separated by commas');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/feature/select`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: features.split(',').map(f => f.trim()) })
        });
        
        const result = await response.json();
        if (result.success) {
            alert(`Selected ${result.count} features`);
            updateFeatureStatus();
        } else {
            alert('Error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function applyPCA() {
    const components = document.getElementById('pcaComponents')?.value || 2;
    
    try {
        const response = await fetch(`${API_BASE}/feature/pca?components=${components}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        if (result.success) {
            alert(`PCA Applied!\nExplained variance: ${(result.total_variance * 100).toFixed(2)}%`);
            updateFeatureStatus();
        } else {
            alert('Error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function restoreFromPCA() {
    try {
        const response = await fetch(`${API_BASE}/feature/restore`, {
            method: 'POST'
        });
        
        const result = await response.json();
        if (result.success) {
            alert('Restored original features');
            updateFeatureStatus();
        } else {
            alert('Error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function keepTopFeatures() {
    const k = prompt('Number of top features to keep:', '10');
    if (!k) return;
    
    try {
        const response = await fetch(`${API_BASE}/feature/keep_top?k=${k}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        if (result.success) {
            alert(`Kept top ${k} features`);
            updateFeatureStatus();
        } else {
            alert('Error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function loadFeatureStatus() {
    try {
        const response = await fetch(`${API_BASE}/feature/status`);
        const result = await response.json();
        
        if (result.success) {
            const statusDiv = document.getElementById('featureStatus');
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = `📌 Using ${result.selected}/${result.total} features`;
        }
    } catch (e) {
        console.error('Feature status error:', e);
    }
}

async function updateFeatureStatus() {
    try {
        const response = await fetch(`${API_BASE}/feature/status`);
        const result = await response.json();
        
        if (result.success) {
            const statusDiv = document.getElementById('featureStatus');
            if (statusDiv) {
                statusDiv.style.display = 'block';
                statusDiv.innerHTML = `📌 Using ${result.selected}/${result.total} features`;
            }
        }
    } catch (e) {
        console.error('Feature status error:', e);
    }
}

// ================================
// Train
// ================================
async function trainModel() {
    const target = document.getElementById('targetColumn').value;
    const model = document.getElementById('modelType').value;
    const testSize = parseFloat(document.getElementById('testSize').value);
    const useSmote = document.getElementById('useSmote').checked;
    const nEstimators = document.getElementById('nEstimators')?.value || 100;
    const maxDepth = document.getElementById('maxDepth')?.value || 10;
    const learningRate = document.getElementById('learningRate')?.value || 0.1;
    
    if (!target) {
        alert('Select a target column');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target: target,
                model: model,
                test_size: testSize,
                use_smote: useSmote,
                n_estimators: parseInt(nEstimators),
                max_depth: parseInt(maxDepth),
                learning_rate: parseFloat(learningRate)
            })
        });
        
        const result = await response.json();
        const resultDiv = document.getElementById('trainResult');
        
        if (result.success) {
            let metricsHtml = '<strong>Metrics:</strong><ul>';
            for (const [k, v] of Object.entries(result.metrics)) {
                if (typeof v !== 'string' && typeof v !== 'object') {
                    metricsHtml += `<li>${k}: ${v.toFixed(4)}</li>`;
                }
            }
            metricsHtml += '</ul>';
            
            resultDiv.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i> Model trained!<br>
                    <strong>Model:</strong> ${result.model}<br>
                    <strong>Problem:</strong> ${result.problem_type}<br>
                    ${metricsHtml}
                </div>
            `;
            updateStatus('Model trained ✓', 'success');
            loadModels();
            loadPredictForm();
        } else {
            resultDiv.innerHTML = `<div class="alert alert-danger">${result.detail || result.message}</div>`;
        }
    } catch (e) {
        alert('Train error: ' + e.message);
    }
}

async function compareAllModels() {
    const target = document.getElementById('targetColumn').value;
    if (!target) {
        alert('Select a target column');
        return;
    }
    
    document.getElementById('comparisonResult').innerHTML = '<div class="alert alert-info"><div class="loading"></div> Training all models...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/train/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target: target })
        });
        
        const result = await response.json();
        
        if (result.success) {
            let html = '<div class="alert alert-success"><strong>🏆 Model Leaderboard</strong><div class="table-responsive mt-2"><table class="table table-sm">';
            html += '<thead><tr><th>Rank</th><th>Model</th><th>Score</th></tr></thead><tbody>';
            result.leaderboard.forEach((m, i) => {
                html += `<tr><td>${i+1}</td><td>${m.model}</td><td>${m.score.toFixed(4)}</td></tr>`;
            });
            html += '</tbody></table></div><strong>Best Model:</strong> ' + result.best_model + '</div>';
            document.getElementById('comparisonResult').innerHTML = html;
            loadModels();
        } else {
            document.getElementById('comparisonResult').innerHTML = `<div class="alert alert-danger">${result.detail}</div>`;
        }
    } catch (e) {
        document.getElementById('comparisonResult').innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
    }
}

function showStackingDialog() {
    fetch(`${API_BASE}/models`)
        .then(r => r.json())
        .then(result => {
            if (result.models.length < 2) {
                alert('Train at least 2 different models first!');
                return;
            }
            
            const models = result.models;
            const modal = document.getElementById('stackingModal');
            const base1 = document.getElementById('stackBase1');
            const base2 = document.getElementById('stackBase2');
            
            base1.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
            base2.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
            if (models.length > 1) base2.value = models[1];
            
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();
        });
}

async function trainStackedModel() {
    const base1 = document.getElementById('stackBase1').value;
    const base2 = document.getElementById('stackBase2').value;
    const meta = document.getElementById('stackMeta').value;
    
    if (base1 === base2) {
        alert('Base models must be different!');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/model/stack`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                base_model_1: base1, 
                base_model_2: base2, 
                meta_model: meta 
            })
        });
        
        const result = await response.json();
        if (result.success) {
            alert(`Stacked model trained!\n\nBase: ${result.base_models.join(' + ')}\nMeta: ${result.meta_model}\nTime: ${result.elapsed}s`);
            const modal = bootstrap.Modal.getInstance(document.getElementById('stackingModal'));
            modal.hide();
            loadModels();
        } else {
            alert('Error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ================================
// Evaluate
// ================================
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const result = await response.json();
        
        if (result.success) {
            const selector = document.getElementById('modelSelector');
            if (selector) {
                selector.innerHTML = result.models.map(m => `<option value="${m}">${m}</option>`).join('');
                if (result.current_model) {
                    selector.value = result.current_model;
                }
            }
            const modelStatus = document.getElementById('modelStatus');
            if (modelStatus && result.current_model) {
                modelStatus.innerHTML = `📌 Current: ${result.current_model}`;
            }
        }
    } catch (e) {
        console.error('Error loading models:', e);
    }
}

async function loadSelectedModel() {
    const modelName = document.getElementById('modelSelector').value;
    if (!modelName) {
        alert('Please select a model');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/model/select`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });
        
        const result = await response.json();
        if (result.success) {
            alert(`Switched to: ${modelName}`);
            updateStatus(`Evaluating: ${modelName}`, 'success');
            document.getElementById('modelStatus').innerHTML = `📌 Current: ${modelName}`;
            await loadMetrics();
        } else {
            alert('Error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Error loading model: ' + e.message);
    }
}

async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE}/evaluate`);
        const result = await response.json();
        
        if (result.success && result.metrics) {
            let html = '<strong>📊 Current Model Metrics</strong><ul>';
            for (const [k, v] of Object.entries(result.metrics)) {
                if (typeof v === 'number') {
                    html += `<li>${k.toUpperCase()}: ${v.toFixed(4)}</li>`;
                }
            }
            html += '</ul>';
            document.getElementById('metricsDisplay').style.display = 'block';
            document.getElementById('metricsDisplay').innerHTML = html;
        }
    } catch (e) {
        console.error('Metrics error:', e);
    }
}

async function generateChart(type) {
    let column = null;
    if (type === 'histogram') {
        column = document.getElementById('chartColumn')?.value;
    }
    
    try {
        const response = await fetch(`${API_BASE}/chart/${type}?column=${column || ''}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            const container = document.getElementById('evalChartContainer');
            const img = document.getElementById('evalChartImage');
            
            img.src = `data:image/png;base64,${result.image}`;
            container.style.display = 'block';
        } else {
            alert('Chart error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Chart error: ' + e.message);
    }
}

async function generateExploreChart() {
    const chartType = document.getElementById('chartType').value;
    const column = document.getElementById('chartColumn')?.value;
    
    try {
        const response = await fetch(`${API_BASE}/chart/${chartType}?column=${column || ''}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            const container = document.getElementById('chartContainer');
            const img = document.getElementById('chartImage');
            
            img.src = `data:image/png;base64,${result.image}`;
            container.style.display = 'block';
        } else {
            alert('Chart error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Chart error: ' + e.message);
    }
}

async function getPermutationImportance() {
    try {
        const response = await fetch(`${API_BASE}/permutation-importance`);
        const result = await response.json();
        
        if (result.success && result.importance && result.importance.length > 0) {
            let html = '<div class="alert alert-info"><strong>🔍 Permutation Importance</strong><ul>';
            for (const item of result.importance.slice(0, 10)) {
                const imp = item.importance !== undefined && item.importance !== null ? item.importance : 0;
                const std = item.std !== undefined && item.std !== null ? item.std : 0;
                html += `<li>${item.feature}: ${imp.toFixed(4)} (std: ${std.toFixed(4)})</li>`;
            }
            html += '</ul></div>';
            document.getElementById('metricsDisplay').innerHTML = html;
            document.getElementById('metricsDisplay').style.display = 'block';
        } else if (result.detail) {
            document.getElementById('metricsDisplay').innerHTML = `<div class="alert alert-warning">${result.detail}</div>`;
            document.getElementById('metricsDisplay').style.display = 'block';
        } else {
            document.getElementById('metricsDisplay').innerHTML = '<div class="alert alert-warning">No permutation importance available. Train a model first.</div>';
            document.getElementById('metricsDisplay').style.display = 'block';
        }
    } catch (e) {
        document.getElementById('metricsDisplay').innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
        document.getElementById('metricsDisplay').style.display = 'block';
    }
}



async function getCrossValidation() {
    try {
        const response = await fetch(`${API_BASE}/cross-validation`);
        const result = await response.json();
        
        if (result.success && result.mean !== undefined) {
            document.getElementById('metricsDisplay').innerHTML = `
                <div class="alert alert-info">
                    <strong>📊 Cross-Validation (${result.cv || 5}-fold)</strong><br>
                    Mean Score: ${result.mean.toFixed(4)}<br>
                    Std Dev: ${result.std.toFixed(4)}<br>
                    Scores: ${result.scores ? result.scores.map(s => s.toFixed(4)).join(', ') : 'N/A'}
                </div>
            `;
            document.getElementById('metricsDisplay').style.display = 'block';
        } else {
            document.getElementById('metricsDisplay').innerHTML = `<div class="alert alert-warning">${result.detail || 'Cross-validation not available'}</div>`;
            document.getElementById('metricsDisplay').style.display = 'block';
        }
    } catch (e) {
        document.getElementById('metricsDisplay').innerHTML = `<div class="alert alert-danger">Error: ${e.message}</div>`;
        document.getElementById('metricsDisplay').style.display = 'block';
    }
}


async function showMisclassified() {
    try {
        const response = await fetch(`${API_BASE}/misclassified`);
        const result = await response.json();
        
        if (result.success && result.samples && result.samples.length > 0) {
            const table = document.getElementById('misclassifiedTable');
            const columns = Object.keys(result.samples[0]);
            table.querySelector('thead').innerHTML = `
                <tr>
                    ${columns.map(col => `<th>${col}</th>`).join('')}
                </tr>
            `;
            table.querySelector('tbody').innerHTML = result.samples.map(row => `
                <tr>
                    ${columns.map(col => {
                        let val = row[col];
                        if (typeof val === 'number') {
                            val = val.toFixed(2);
                        }
                        return `<td>${val !== undefined && val !== null ? val : ''}</td>`;
                    }).join('')}
                </tr>
            `).join('');
            
            const modal = new bootstrap.Modal(document.getElementById('misclassifiedModal'));
            modal.show();
        } else if (result.detail) {
            alert(result.detail);
        } else {
            alert('No misclassified samples found or model not trained');
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function getFeatureImportance() {
    try {
        const response = await fetch(`${API_BASE}/feature-importance-plot`);
        const result = await response.json();
        
        if (result.success) {
            const container = document.getElementById('evalChartContainer');
            const img = document.getElementById('evalChartImage');
            img.src = `data:image/png;base64,${result.image}`;
            container.style.display = 'block';
        } else {
            alert('Error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ================================
// Predict
// ================================
async function loadPredictForm() {
    try {
        const response = await fetch(`${API_BASE}/input-schema`);
        const result = await response.json();
        
        if (result.success && result.schema) {
            const formDiv = document.getElementById('predictForm');
            if (result.schema.length === 0) {
                formDiv.innerHTML = '<div class="alert alert-warning">No features available. Train a model first.</div>';
            } else {
                formDiv.innerHTML = result.schema.map(field => {
                    if (field.type === 'categorical') {
                        return `
                            <div class="mb-3">
                                <label class="form-label">${field.name}</label>
                                <select class="form-select" data-feature="${field.name}">
                                    ${field.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}
                                </select>
                            </div>
                        `;
                    } else {
                        return `
                            <div class="mb-3">
                                <label class="form-label">${field.name}</label>
                                <input type="number" step="any" class="form-control" data-feature="${field.name}" value="${field.mean || 0}">
                            </div>
                        `;
                    }
                }).join('');
            }
        } else {
            document.getElementById('predictForm').innerHTML = '<div class="alert alert-warning">Train a model first to enable predictions</div>';
        }
    } catch (e) {
        console.error('Predict form error:', e);
        document.getElementById('predictForm').innerHTML = '<div class="alert alert-danger">Error loading prediction form</div>';
    }
}

async function makePrediction() {
    const inputs = {};
    document.querySelectorAll('#predictForm input, #predictForm select').forEach(field => {
        inputs[field.dataset.feature] = field.value;
    });
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputs)
        });
        
        const result = await response.json();
        const resultDiv = document.getElementById('predictResult');
        
        if (result.success) {
            let html = `
                <div class="alert alert-success">
                    <strong>🔮 Prediction:</strong> ${result.result.prediction}<br>
            `;
            if (result.result.confidence) {
                html += `<strong>Confidence:</strong> ${(result.result.confidence * 100).toFixed(2)}%<br>`;
            }
            if (result.result.probabilities) {
                html += `<strong>Probabilities:</strong><ul>`;
                for (const [cls, prob] of Object.entries(result.result.probabilities)) {
                    html += `<li>${cls}: ${(prob * 100).toFixed(2)}%</li>`;
                }
                html += `</ul>`;
            }
            html += `</div>`;
            resultDiv.innerHTML = html;
        } else {
            resultDiv.innerHTML = `<div class="alert alert-danger">${result.detail || result.message}</div>`;
        }
    } catch (e) {
        alert('Predict error: ' + e.message);
    }
}

// ================================
// Insights
// ================================
async function generateInsights() {
    try {
        const response = await fetch(`${API_BASE}/insights`);
        const result = await response.json();
        
        if (result.success) {
            const list = document.getElementById('insightsList');
            list.innerHTML = result.insights.map(insight => `<li>${insight}</li>`).join('');
        } else {
            alert('Error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Insights error: ' + e.message);
    }
}

async function showHealthScore() {
    try {
        const response = await fetch(`${API_BASE}/health-score`);
        const result = await response.json();
        
        if (result.success) {
            const container = document.getElementById('healthScoreContainer');
            const gradeClass = result.grade;
            container.style.display = 'block';
            container.innerHTML = `
                <div class="health-score ${gradeClass}">
                    <h2>${result.score}/100</h2>
                    <p>Grade: ${result.grade}</p>
                    <small>${result.issues.length} issues detected</small>
                </div>
            `;
        } else {
            alert('Error: ' + (result.detail || result.message));
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function exportReport() {
    try {
        window.location.href = `${API_BASE}/export/report`;
    } catch (e) {
        alert('Export error: ' + e.message);
    }
}

// ================================
// Helpers
// ================================
function updateStatus(text, type) {
    const status = document.getElementById('status');
    const badgeClass = type === 'success' ? 'badge-success' : 'badge-warning';
    if (status) {
        status.innerHTML = `<span class="badge-status ${badgeClass}">${text}</span>`;
    }
}

// Load initial status
async function init() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const result = await response.json();
        if (result.loaded) {
            updateStatus(`Loaded: ${result.rows} rows`, 'success');
            loadColumns();
            loadModels();
        }
    } catch (e) {
        console.error('Init error:', e);
    }
}

init();