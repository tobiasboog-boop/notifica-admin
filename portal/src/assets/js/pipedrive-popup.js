/**
 * Pipedrive Web Forms Popup Handler
 * Handles opening/closing of Pipedrive forms in popup modals
 */

(function() {
  "use strict";

  // Pipedrive form configurations
  // Replace these URLs with your actual Pipedrive Web Form URLs
  const PIPEDRIVE_FORMS = {
    demo: {
      title: "Plan een demo",
      subtitle: "Ik wil graag een demo sessie plannen",
      url: "https://webforms.pipedrive.com/f/czAN5Djwx6GKDvEFyOO9B3HhfpFEeQ8NX7pcdOLeHJ5LSIQEP8IzJRQE6kjvn2LncD",
      defaultMessage: "Ik wil graag een demo sessie plannen."
    },
    contact: {
      title: "Neem contact op",
      subtitle: "Vul onderstaand formulier in en we nemen binnen 24 uur contact met je op.",
      url: "https://webforms.pipedrive.com/f/6zmGQ2X1XW91oGi0OQQ0vma5UHXZGOy6qV6Z5KEIv8DppStJUhO0CP4sMxRkcUl0LV"
    },
    whitepaper: {
      title: "Download whitepaper",
      subtitle: "Geef aan welke whitepapers je wenst te ontvangen in onderstaand bericht.",
      url: "https://webforms.pipedrive.com/f/czAN5Djwx6GKDvEFyOO9B3HhfpFEeQ8NX7pcdOLeHJ5LSIQEP8IzJRQE6kjvn2LncD",
      defaultMessage: "Stuur mij whitepaper: {title}"
    },
    support: {
      title: "Remote support",
      subtitle: "Ik heb hulp nodig met een technisch probleem",
      url: "https://webforms.pipedrive.com/f/czAN5Djwx6GKDvEFyOO9B3HhfpFEeQ8NX7pcdOLeHJ5LSIQEP8IzJRQE6kjvn2LncD",
      defaultMessage: "Ik heb hulp nodig met een technisch probleem."
    },
    training: {
      title: "Training aanvragen",
      subtitle: "Vul onderstaand formulier in en we nemen contact met je op over beschikbare trainingsdagen.",
      url: "https://webforms.pipedrive.com/f/6zmGQ2X1XW91oGi0OQQ0vma5UHXZGOy6qV6Z5KEIv8DppStJUhO0CP4sMxRkcUl0LV"
    }
  };

  // Create popup overlay element
  function createPopupOverlay() {
    const overlay = document.createElement("div");
    overlay.className = "popup-overlay";
    overlay.id = "pipedrive-popup";
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
  function openPopup(formId, context = {}) {
    const form = PIPEDRIVE_FORMS[formId];
    if (!form) {
      console.warn(`Pipedrive form "${formId}" not found`);
      return;
    }

    let overlay = document.getElementById("pipedrive-popup");
    if (!overlay) {
      overlay = createPopupOverlay();
    }

    const container = document.getElementById("pipedrive-form-container");

    // Load Pipedrive form using their embed method
    container.innerHTML = `
      ${form.title ? `<h2 style="margin-bottom: ${form.subtitle ? "12px" : "24px"}; color: #16136F;">${form.title}</h2>` : ""}
      ${form.subtitle ? `<p style="margin-bottom: 24px; color: #6B7280; font-size: 15px;">${form.subtitle}</p>` : ""}
      <div class="pipedriveWebForms" data-pd-webforms="${form.url}"></div>
    `;

    // Load Pipedrive script
    const existingScript = document.querySelector("script[src='https://webforms.pipedrive.com/f/loader']");
    if (existingScript) {
      existingScript.remove();
    }

    const script = document.createElement("script");
    script.src = "https://webforms.pipedrive.com/f/loader";
    container.appendChild(script);

    // Show overlay
    overlay.classList.add("active");
    document.body.style.overflow = "hidden";

    // Focus trap
    overlay.focus();
  }

  // Close popup
  function closePopup() {
    const overlay = document.getElementById("pipedrive-popup");
    if (overlay) {
      overlay.classList.remove("active");
      document.body.style.overflow = "";

      // Clear form container
      const container = document.getElementById("pipedrive-form-container");
      if (container) {
        container.innerHTML = "";
      }
    }
  }

  // Initialize event listeners
  function init() {
    // Handle button clicks
    document.addEventListener("click", function(e) {
      const button = e.target.closest("[data-pipedrive-form]");
      if (button) {
        e.preventDefault();
        const formId = button.getAttribute("data-pipedrive-form");

        // Get context data from button attributes
        const context = {
          title: button.getAttribute("data-title") || ""
        };

        openPopup(formId, context);
      }

      // Close button
      if (e.target.closest(".popup-close")) {
        closePopup();
      }

      // Click outside popup content
      if (e.target.classList.contains("popup-overlay")) {
        closePopup();
      }
    });

    // Escape key to close
    document.addEventListener("keydown", function(e) {
      if (e.key === "Escape") {
        closePopup();
      }
    });
  }

  // Initialize when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Expose functions globally for external use
  window.NotificaPopup = {
    open: openPopup,
    close: closePopup
  };

  // ==========================================================================
  // SUCCESS MESSAGE OVERLAY
  // Monitors Pipedrive forms and shows a styled success message
  // ==========================================================================

  function createSuccessOverlay(container) {
    const overlay = document.createElement("div");
    overlay.className = "pipedrive-success-overlay";
    overlay.innerHTML = `
      <div class="pipedrive-success-content">
        <div class="pipedrive-success-icon">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
            <polyline points="22 4 12 14.01 9 11.01"/>
          </svg>
        </div>
        <h3>Je bericht is verzonden!</h3>
        <p>Bedankt voor je bericht. We nemen binnen 24 uur contact met je op.</p>
      </div>
    `;
    container.appendChild(overlay);
    return overlay;
  }

  function showSuccessOverlay(container) {
    let overlay = container.querySelector(".pipedrive-success-overlay");
    if (!overlay) {
      overlay = createSuccessOverlay(container);
    }
    // Small delay for smooth appearance
    requestAnimationFrame(() => {
      overlay.classList.add("active");
    });
  }

  // Monitor for Pipedrive success state
  function monitorPipedriveSuccess() {
    // Use MutationObserver to detect when Pipedrive shows success message
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        // Check for text content changes that indicate success
        if (mutation.type === "childList" || mutation.type === "characterData") {
          const target = mutation.target;

          // Look for success indicators in the DOM
          const containers = document.querySelectorAll(".pipedriveWebForms");
          containers.forEach((container) => {
            // Check if container or its children contain success text
            const text = container.textContent || "";
            if (text.includes("verzonden") || text.includes("Thank you") || text.includes("Bedankt")) {
              // Find the parent form-container or use the container itself
              const formContainer = container.closest(".form-container") || container.parentElement;
              if (formContainer && !formContainer.querySelector(".pipedrive-success-overlay.active")) {
                showSuccessOverlay(formContainer);
              }
            }
          });
        }
      });
    });

    // Start observing the document for changes
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true
    });
  }

  // Start monitoring when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", monitorPipedriveSuccess);
  } else {
    monitorPipedriveSuccess();
  }
})();
