/**
 * Lightweight Cookie Consent for Notifica
 * AVG/GDPR compliant - loads analytics only after consent
 */
(function() {
  'use strict';

  const CONSENT_KEY = 'notifica_cookie_consent';
  const CONSENT_VERSION = '1'; // Increment to re-ask consent after policy changes

  // Check if consent was already given
  function getConsent() {
    try {
      const consent = localStorage.getItem(CONSENT_KEY);
      if (consent) {
        const parsed = JSON.parse(consent);
        if (parsed.version === CONSENT_VERSION) {
          return parsed;
        }
      }
    } catch (e) {}
    return null;
  }

  // Save consent choice
  function setConsent(analytics) {
    const consent = {
      version: CONSENT_VERSION,
      analytics: analytics,
      timestamp: new Date().toISOString()
    };
    localStorage.setItem(CONSENT_KEY, JSON.stringify(consent));
    return consent;
  }

  // Load Google Analytics
  function loadAnalytics() {
    if (window.gaLoaded) return;
    window.gaLoaded = true;

    // Load gtag script
    const script = document.createElement('script');
    script.async = true;
    script.src = 'https://www.googletagmanager.com/gtag/js?id=G-X93YGT3NLM';
    document.head.appendChild(script);

    // Initialize gtag
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    window.gtag = gtag;
    gtag('js', new Date());
    gtag('config', 'G-X93YGT3NLM');
  }

  // Note: Pipedrive LeadFeeder is loaded directly in base-notion.njk
  // (not behind cookie consent - B2B lead tracking under legitimate interest)

  // Create and show cookie banner
  function showBanner() {
    const banner = document.createElement('div');
    banner.id = 'cookie-banner';
    banner.innerHTML = `
      <div class="cookie-banner-content">
        <div class="cookie-banner-text">
          <strong>Cookies</strong>
          <p>Wij gebruiken cookies om onze website te verbeteren. <a href="/cookies/">Meer informatie</a></p>
        </div>
        <div class="cookie-banner-buttons">
          <button id="cookie-reject" class="cookie-btn cookie-btn-secondary">Alleen noodzakelijk</button>
          <button id="cookie-accept" class="cookie-btn cookie-btn-primary">Accepteren</button>
        </div>
      </div>
    `;
    document.body.appendChild(banner);

    // Add event listeners
    document.getElementById('cookie-accept').addEventListener('click', function() {
      setConsent(true);
      loadAnalytics();
      hideBanner();
    });

    document.getElementById('cookie-reject').addEventListener('click', function() {
      setConsent(false);
      hideBanner();
    });

    // Animate in
    requestAnimationFrame(function() {
      banner.classList.add('cookie-banner-visible');
    });
  }

  // Hide cookie banner
  function hideBanner() {
    const banner = document.getElementById('cookie-banner');
    if (banner) {
      banner.classList.remove('cookie-banner-visible');
      setTimeout(function() {
        banner.remove();
      }, 300);
    }
  }

  // Open cookie settings (can be called from footer link)
  window.openCookieSettings = function() {
    // Remove existing consent
    localStorage.removeItem(CONSENT_KEY);
    // Remove existing banner if any
    hideBanner();
    // Show new banner
    setTimeout(showBanner, 100);
  };

  // Initialize on DOM ready
  function init() {
    const consent = getConsent();

    if (consent === null) {
      // No consent given yet, show banner
      showBanner();
    } else if (consent.analytics) {
      // Consent given, load analytics
      loadAnalytics();
    }
    // If consent.analytics is false, do nothing (user rejected)
  }

  // Run init when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
