// Debug script to force testimonials grid to 3 columns
// Run this in browser console to test

document.addEventListener('DOMContentLoaded', function() {
    const grid = document.querySelector('.testimonials-grid');

    if (grid) {
        console.log('Found testimonials-grid');
        console.log('Current styles:', window.getComputedStyle(grid));

        // Force the grid layout with maximum specificity
        grid.style.cssText = 'display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 20px !important;';

        console.log('After force:', window.getComputedStyle(grid).gridTemplateColumns);
    } else {
        console.log('testimonials-grid NOT FOUND');
    }
});
