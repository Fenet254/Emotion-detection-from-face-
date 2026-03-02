/**
 * Emotion AI - Main JavaScript
 * Handles common functionality across all pages
 */

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Emotion AI initialized');
    
    // Update model status indicator
    updateModelStatus();
    
    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});

/**
 * Update model status indicator
 */
async function updateModelStatus() {
    try {
        const response = await fetch('/api/model/status');
        const data = await response.json();
        
        const statusIndicator = document.getElementById('modelStatus');
        if (statusIndicator) {
            const statusDot = statusIndicator.querySelector('.status-dot');
            const statusText = statusIndicator.querySelector('.status-text');
            
            if (data.loaded) {
                statusIndicator.classList.add('loaded');
                if (statusDot) statusDot.style.background = '#10b981';
                if (statusText) statusText.textContent = 'Model Loaded';
            } else {
                statusIndicator.classList.remove('loaded');
                if (statusDot) statusDot.style.background = '#ef4444';
                if (statusText) statusText.textContent = 'Model Not Loaded';
            }
        }
    } catch (err) {
        console.error('Error fetching model status:', err);
    }
}

/**
 * Format confidence percentage
 */
function formatConfidence(value) {
    return (value * 100).toFixed(1) + '%';
}

/**
 * Format date/time
 */
function formatDateTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Create emotion bar chart
 */
function createEmotionBars(container, emotions, maxWidth = 100) {
    container.innerHTML = '';
    
    // Sort by percentage
    const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
    
    sorted.forEach(([emotion, percent]) => {
        const bar = document.createElement('div');
        bar.className = 'emotion-bar';
        bar.innerHTML = `
            <span class="emotion-bar-label">${emotion}</span>
            <div class="emotion-bar-track">
                <div class="emotion-bar-fill" style="width: ${percent}%"></div>
            </div>
            <span class="emotion-bar-value">${percent.toFixed(1)}%</span>
        `;
        container.appendChild(bar);
    });
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#6366f1'};
        color: white;
        border-radius: 8px;
        z-index: 9999;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

/**
 * Add animation keyframes dynamically
 */
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    @keyframes slideOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(20px);
        }
    }
`;
document.head.appendChild(style);

// Export for use in other scripts
window.EmotionAI = {
    updateModelStatus,
    formatConfidence,
    formatDateTime,
    createEmotionBars,
    showNotification
};
