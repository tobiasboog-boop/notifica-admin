/**
 * Mobile Navigation Handler
 * Handles the hamburger menu toggle for mobile devices
 * and header scroll behavior
 */

(function() {
  'use strict';

  const navToggle = document.querySelector('.nav-toggle');
  const navMenu = document.querySelector('.nav-menu');
  const header = document.querySelector('.header');

  // Header scroll behavior - shrink logo on scroll
  if (header) {
    window.addEventListener('scroll', function() {
      if (window.scrollY > 50) {
        header.classList.add('scrolled');
      } else {
        header.classList.remove('scrolled');
      }
    }, { passive: true });
  }

  if (!navToggle || !navMenu) return;

  function openMenu() {
    navToggle.setAttribute('aria-expanded', 'true');
    navToggle.classList.add('active');
    navMenu.classList.add('active');
    header.classList.add('menu-open');

    // Lock body scroll - use simpler approach
    document.body.style.overflow = 'hidden';
    document.body.style.touchAction = 'none';
  }

  function closeMenu() {
    navToggle.setAttribute('aria-expanded', 'false');
    navToggle.classList.remove('active');
    navMenu.classList.remove('active');
    header.classList.remove('menu-open');

    // Close all dropdowns when menu closes
    document.querySelectorAll('.nav-item.dropdown-open').forEach(function(item) {
      item.classList.remove('dropdown-open');
    });

    // Restore body scroll
    document.body.style.overflow = '';
    document.body.style.touchAction = '';
  }

  // Toggle mobile menu
  navToggle.addEventListener('click', function(e) {
    e.stopPropagation();
    const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';

    if (isExpanded) {
      closeMenu();
    } else {
      openMenu();
    }
  });

  // Allow touch scrolling inside the menu
  navMenu.addEventListener('touchmove', function(e) {
    e.stopPropagation();
  }, { passive: true });

  // Close menu when clicking on a dropdown link (actual page navigation)
  navMenu.querySelectorAll('.dropdown-link').forEach(function(link) {
    link.addEventListener('click', function() {
      closeMenu();
    });
  });

  // Close menu when clicking on the overlay (not the menu itself)
  header.addEventListener('click', function(e) {
    if (header.classList.contains('menu-open') &&
        !navMenu.contains(e.target) &&
        !navToggle.contains(e.target)) {
      closeMenu();
    }
  });

  // Handle dropdown toggles on mobile
  const dropdownItems = document.querySelectorAll('.nav-item.has-dropdown');

  dropdownItems.forEach(function(item) {
    const link = item.querySelector('.nav-link');

    link.addEventListener('click', function(e) {
      // Only handle on mobile
      if (window.innerWidth <= 1024) {
        e.preventDefault();
        e.stopPropagation();

        // Close other dropdowns first
        dropdownItems.forEach(function(otherItem) {
          if (otherItem !== item && otherItem.classList.contains('dropdown-open')) {
            otherItem.classList.remove('dropdown-open');
          }
        });

        // Toggle this dropdown
        item.classList.toggle('dropdown-open');
      }
    });
  });

  // Close menu on escape key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && navMenu.classList.contains('active')) {
      closeMenu();
    }
  });

  // Reset on window resize
  window.addEventListener('resize', function() {
    if (window.innerWidth > 1024 && navMenu.classList.contains('active')) {
      closeMenu();
    }
  });

  // Prevent body scroll when touching outside menu but inside overlay
  document.addEventListener('touchmove', function(e) {
    if (header.classList.contains('menu-open') && !navMenu.contains(e.target)) {
      e.preventDefault();
    }
  }, { passive: false });
})();
