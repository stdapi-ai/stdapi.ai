/*
 * Homepage enhancements (docs/index.md): the logo marquee and the
 * differentiator carousel.
 *
 * Progressive enhancement — without JavaScript the marquee still scrolls (CSS
 * animation) and every carousel panel renders stacked with no dead controls.
 * With JavaScript the marquee track is doubled for a seamless loop and the
 * carousel shows one panel at a time.
 *
 * Loaded via extra_javascript, so it runs on every page. Two small site-wide
 * accessibility repairs ride along here because this is the only such hook:
 * naming Material's progress bar and tab-stopping scrollable code blocks.
 */
(function () {
  "use strict";

  //: Auto-advance interval, in milliseconds.
  var ADVANCE_MS = 7000;

  //: Teardown callbacks for the previous page render (Material instant nav).
  var teardowns = [];

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* Duplicate the track so the CSS translateX(-50%) loop is seamless. The
     clones repeat existing links, so they are hidden from assistive tech and
     removed from the tab order. */
  function setupMarquee(track) {
    if (track.dataset.duplicated) return;
    var fragment = document.createDocumentFragment();
    var items = track.children;
    for (var i = 0; i < items.length; i++) {
      var clone = items[i].cloneNode(true);
      clone.setAttribute("aria-hidden", "true");
      clone.setAttribute("tabindex", "-1");
      fragment.appendChild(clone);
    }
    track.appendChild(fragment);
    track.dataset.duplicated = "1";
  }

  function setupCarousel(root) {
    var panels = [].slice.call(root.querySelectorAll(".carousel__panel"));
    if (panels.length < 2) return;

    var nav = document.createElement("div");
    nav.className = "carousel__nav";
    var tablist = document.createElement("div");
    tablist.className = "carousel__tabs";
    tablist.setAttribute("role", "tablist");
    tablist.setAttribute("aria-label", "Differentiators");
    var dots = document.createElement("div");
    dots.className = "carousel__dots";

    var tabs = [];
    var dotButtons = [];

    panels.forEach(function (panel, i) {
      var label = panel.getAttribute("data-tab") || "Panel " + (i + 1);
      panel.id = "home-carousel-panel-" + i;
      panel.setAttribute("role", "tabpanel");
      panel.setAttribute("aria-labelledby", "home-carousel-tab-" + i);

      var tab = document.createElement("button");
      tab.type = "button";
      tab.id = "home-carousel-tab-" + i;
      tab.className = "carousel__tab";
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", panel.id);
      tab.textContent = label;
      tablist.appendChild(tab);
      tabs.push(tab);

      /* The dots duplicate the tabs' function, so they are a pointer-only
         affordance: hidden from AT and skipped by the keyboard. */
      var dot = document.createElement("button");
      dot.type = "button";
      dot.className = "carousel__dot";
      dot.tabIndex = -1;
      dot.setAttribute("aria-hidden", "true");
      dots.appendChild(dot);
      dotButtons.push(dot);
    });

    nav.appendChild(tablist);
    nav.appendChild(dots);
    root.insertBefore(nav, root.firstChild);
    root.classList.add("carousel--js");

    var current = -1;

    function select(index) {
      if (index === current) return;
      current = index;
      for (var i = 0; i < panels.length; i++) {
        var active = i === index;
        panels[i].hidden = !active;
        tabs[i].setAttribute("aria-selected", active ? "true" : "false");
        // Only the selected tab is tabbable; arrow keys move between them.
        tabs[i].tabIndex = active ? 0 : -1;
        dotButtons[i].setAttribute("aria-selected", active ? "true" : "false");
      }
    }

    // Auto-advance stops for good once the reader picks a panel themselves.
    var timer = null;
    var stopped = reduceMotion.matches;
    var held = 0;

    function tick() {
      if (!held && !document.hidden) select((current + 1) % panels.length);
    }

    function stop() {
      stopped = true;
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    }

    function pick(index) {
      stop();
      select(index);
    }

    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () {
        pick(i);
      });
      tab.addEventListener("keydown", function (event) {
        var next;
        if (event.key === "ArrowRight") next = (i + 1) % panels.length;
        else if (event.key === "ArrowLeft")
          next = (i - 1 + panels.length) % panels.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = panels.length - 1;
        else return;
        event.preventDefault();
        pick(next);
        tabs[next].focus();
      });
    });

    dotButtons.forEach(function (dot, i) {
      dot.addEventListener("click", function () {
        pick(i);
      });
    });

    // Pause for pointer and keyboard users alike (WCAG 2.2.2).
    function hold() {
      held++;
    }
    function release() {
      held = Math.max(0, held - 1);
    }
    var holds = [
      ["mouseenter", hold],
      ["mouseleave", release],
      ["focusin", hold],
      ["focusout", release],
    ];
    holds.forEach(function (pair) {
      root.addEventListener(pair[0], pair[1]);
    });

    function onMotionChange() {
      if (reduceMotion.matches) stop();
    }
    reduceMotion.addEventListener("change", onMotionChange);

    select(0);
    if (!stopped) timer = setInterval(tick, ADVANCE_MS);

    teardowns.push(function () {
      stop();
      holds.forEach(function (pair) {
        root.removeEventListener(pair[0], pair[1]);
      });
      reduceMotion.removeEventListener("change", onMotionChange);
    });
  }

  /* Material injects its instant-navigation progress bar with role=progressbar
     but no accessible name; this script is the only site-wide hook we have. */
  function nameProgressBar() {
    var bar = document.querySelector(".md-progress:not([aria-label])");
    if (bar) bar.setAttribute("aria-label", "Loading page");
  }

  /* A code block that scrolls sideways must be reachable by keyboard
     (WCAG 2.1.1). Only the blocks that actually overflow become tab stops.
     Measure everything before writing, so the loop cannot thrash layout. */
  function tabStopScrollableCode() {
    var blocks = document.querySelectorAll(".md-typeset pre > code");
    var overflowing = [];
    for (var i = 0; i < blocks.length; i++) {
      overflowing.push(blocks[i].scrollWidth > blocks[i].clientWidth);
    }
    for (var j = 0; j < blocks.length; j++) {
      if (overflowing[j]) blocks[j].setAttribute("tabindex", "0");
      else blocks[j].removeAttribute("tabindex");
    }
  }

  /* The ReDoc plugin embeds the API reference in an untitled iframe. */
  function nameRedocFrame() {
    var frame = document.querySelector("iframe.redoc-iframe:not([title])");
    if (frame) frame.setAttribute("title", "API reference");
  }

  var resizeTimer = null;
  window.addEventListener(
    "resize",
    function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(tabStopScrollableCode, 200);
    },
    { passive: true }
  );

  function init() {
    teardowns.splice(0).forEach(function (fn) {
      fn();
    });
    nameProgressBar();
    nameRedocFrame();
    tabStopScrollableCode();
    [].forEach.call(document.querySelectorAll(".logo-track"), setupMarquee);
    [].forEach.call(document.querySelectorAll("[data-carousel]"), setupCarousel);
  }

  // Material's instant navigation re-emits document$ on every page swap.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
