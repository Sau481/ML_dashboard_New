// ========================================
// ML Dashboard - JavaScript Utilities
// ========================================

/**
 * Utility Functions
 */

// Debounce function for performance optimization
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Validate CSV file
function validateCSVFile(file) {
    const validTypes = ['text/csv', 'application/vnd.ms-excel'];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (!file) {
        return { valid: false, error: 'No file selected' };
    }

    if (!file.name.endsWith('.csv') && !validTypes.includes(file.type)) {
        return { valid: false, error: 'Please select a CSV file' };
    }

    if (file.size > maxSize) {
        return { valid: false, error: 'File size exceeds 50MB limit' };
    }

    return { valid: true };
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = 'flash';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        animation: slideInRight 0.3s ease;
    `;

    const colors = {
        success: '#00f2fe',
        error: '#f5576c',
        warning: '#fee140',
        info: '#667eea'
    };

    notification.style.borderLeftColor = colors[type] || colors.info;
    notification.innerHTML = `<strong>${message}</strong>`;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Smooth scroll to element
function smoothScrollTo(element) {
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Create loading overlay
function createLoadingOverlay(message = 'Processing...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loadingOverlay';
    overlay.innerHTML = `
        <div style="text-align: center;">
            <div class="loading-spinner"></div>
            <p style="margin-top: var(--spacing-md); font-size: 1.25rem; font-weight: 600;">
                ${message}
            </p>
        </div>
    `;
    document.body.appendChild(overlay);
    return overlay;
}

// Remove loading overlay
function removeLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Form Validation
 */

// Validate form inputs
function validateForm(formElement) {
    const inputs = formElement.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            input.style.borderColor = 'var(--accent-color)';

            // Reset border color after 2 seconds
            setTimeout(() => {
                input.style.borderColor = '';
            }, 2000);
        }
    });

    return isValid;
}

/**
 * Table Utilities
 */

// Make table sortable
function makeSortable(table) {
    const headers = table.querySelectorAll('th');
    headers.forEach((header, index) => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => sortTable(table, index));
    });
}

// Sort table by column
function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    const sortedRows = rows.sort((a, b) => {
        const aValue = a.cells[columnIndex].textContent.trim();
        const bValue = b.cells[columnIndex].textContent.trim();

        // Try numeric comparison first
        const aNum = parseFloat(aValue);
        const bNum = parseFloat(bValue);

        if (!isNaN(aNum) && !isNaN(bNum)) {
            return aNum - bNum;
        }

        // Fall back to string comparison
        return aValue.localeCompare(bValue);
    });

    // Re-append sorted rows
    sortedRows.forEach(row => tbody.appendChild(row));
}

/**
 * Animation Utilities
 */

// Animate number counting
function animateNumber(element, start, end, duration = 1000) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current * 10000) / 10000;
    }, 16);
}

// Fade in element
function fadeIn(element, duration = 300) {
    element.style.opacity = 0;
    element.style.display = 'block';

    let opacity = 0;
    const increment = 16 / duration;

    const timer = setInterval(() => {
        opacity += increment;
        if (opacity >= 1) {
            opacity = 1;
            clearInterval(timer);
        }
        element.style.opacity = opacity;
    }, 16);
}

// Fade out element
function fadeOut(element, duration = 300) {
    let opacity = 1;
    const decrement = 16 / duration;

    const timer = setInterval(() => {
        opacity -= decrement;
        if (opacity <= 0) {
            opacity = 0;
            element.style.display = 'none';
            clearInterval(timer);
        }
        element.style.opacity = opacity;
    }, 16);
}

/**
 * Chart Utilities
 */

// Common Chart.js configuration
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
        legend: {
            display: true,
            position: 'top',
            labels: {
                font: {
                    family: 'Inter, sans-serif',
                    size: 12
                },
                padding: 15
            }
        }
    },
    scales: {
        y: {
            beginAtZero: true,
            grid: {
                color: 'rgba(0, 0, 0, 0.05)'
            },
            ticks: {
                font: {
                    family: 'Inter, sans-serif'
                }
            }
        },
        x: {
            grid: {
                display: false
            },
            ticks: {
                font: {
                    family: 'Inter, sans-serif'
                }
            }
        }
    }
};

// Color palette for charts
const chartColors = {
    primary: 'rgba(102, 126, 234, 0.8)',
    primaryBorder: 'rgba(102, 126, 234, 1)',
    accent: 'rgba(245, 87, 108, 0.8)',
    accentBorder: 'rgba(245, 87, 108, 1)',
    success: 'rgba(0, 242, 254, 0.8)',
    successBorder: 'rgba(0, 242, 254, 1)',
    warning: 'rgba(254, 225, 64, 0.8)',
    warningBorder: 'rgba(254, 225, 64, 1)',
    purple: 'rgba(118, 75, 162, 0.8)',
    purpleBorder: 'rgba(118, 75, 162, 1)'
};

/**
 * Export utilities
 */
window.MLDashboard = {
    debounce,
    formatFileSize,
    validateCSVFile,
    showNotification,
    smoothScrollTo,
    createLoadingOverlay,
    removeLoadingOverlay,
    validateForm,
    makeSortable,
    sortTable,
    animateNumber,
    fadeIn,
    fadeOut,
    chartDefaults,
    chartColors
};
