/**
 * Pipedrive Web Forms Popup Handler
 * Handles opening/closing of Pipedrive forms in popup modals
 */

(function() {
  'use strict';

  // Pipedrive form configurations
  // Replace these URLs with your actual Pipedrive Web Form URLs
  const PIPEDRIVE_FORMS = {
    demo: {
      title: 'Plan een demo',
      url: 'https://webforms.pipedrive.com/f/czAN5Djwx6GKDvEFyOO9B3HhfpFEeQ8NX7pcdOLeHJ5LSIQEP8IzJRQE6kjvn2LncD'
    },
    contact: {
      title: 'Neem contact op',
      url: 'https://webforms.pipedrive.com/f/6zmGQ2X1XW91oGi0OQQ0vma5UHXZGOy6qV6Z5KEIv8DppStJUhO0CP4sMxRkcUl0LV'
    },
    whitepaper: {
      title: 'Download whitepaper',
      url: 'https://webforms.pipedrive.com/f/czAN5Djwx6GKDvEFyOO9B3HhfpFEeQ8NX7pcdOLeHJ5LSIQEP8IzJRQE6kjvn2LncD'
    },
    support: {
      title: 'Remote support',
      url: 'https://webforms.pipedrive.com/f/czAN5Djwx6GKDvEFyOO9B3HhfpFEeQ8NX7pcdOLeHJ5LSIQEP8IzJRQE6kjvn2LncD'
    }
  };

  // Create popup overlay element
  function createPopupOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'popup-overlay';
    overlay.id = 'pipedrive-popup';
    overlay.innerHTML = `
      <div class="popup-content">
        <button class="popup-close" aria-label="Sluiten">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6L6 18M6 6l12 12"/>
          </svg>
        </button>
        <div class="popup-body" id="pipedrive-form-container">
          <!-- Pipedrive form will be loaded here -->
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    return overlay;
  }

  // Open popup with specific form
  function openPopup(formId) {
    const form = PIPEDRIVE_FORMS[formId];
    if (!form) {
      console.warn(`Pipedrive form "${formId}" not found`);
      return;
    }

    let overlay = document.getElementById('pipedrive-popup');
    if (!overlay) {
      overlay = createPopupOverlay();
    }

    const container = document.getElementById('pipedrive-form-container');

    // Load Pipedrive form using their embed method
    container.innerHTML = `
      <div class="pipedriveWebForms" data-pd-webforms="${form.url}"></div>
    `;

    // Load Pipedrive script
    const existingScript = document.querySelector('script[src="https://webforms.pipedrive.com/f/loader"]');
    if (existingScript) {
      existingScript.remove();
    }

    const script = document.createElement('script');
    script.src = 'https://webforms.pipedrive.com/f/loader';
    container.appendChild(script);

    // Show overlay
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Focus trap
    overlay.focus();
  }

  // Close popup
  function closePopup() {
    const overlay = document.getElementById('pipedrive-popup');
    if (overlay) {
      overlay.classList.remove('active');
      document.body.style.overflow = '';

      // Clear form container
      const container = document.getElementById('pipedrive-form-container');
      if (container) {
        container.innerHTML = '';
      }
    }
  }

  // Initialize event listeners
  function init() {
    // Handle button clicks
    document.addEventListener('click', function(e) {
      const button = e.target.closest('[data-pipedrive-form]');
      if (button) {
        e.preventDefault();
        const formId = button.getAttribute('data-pipedrive-form');
        openPopup(formId);
      }

      // Close button
      if (e.target.closest('.popup-close')) {
        closePopup();
      }

      // Click outside popup content
      if (e.target.classList.contains('popup-overlay')) {
        closePopup();
      }
    });

    // Escape key to close
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        closePopup();
      }
    });
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose functions globally for external use
  window.NotificaPopup = {
    open: openPopup,
    close: closePopup
  };
})();
