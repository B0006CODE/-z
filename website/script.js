const progress = document.querySelector("#scroll-progress");
const tabs = document.querySelectorAll(".tab-button");
const tableViews = document.querySelectorAll(".table-view");

function updateProgress() {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const value = scrollable > 0 ? window.scrollY / scrollable : 0;
  progress.style.width = `${Math.min(value * 100, 100)}%`;
}

window.addEventListener("scroll", updateProgress, { passive: true });
updateProgress();

tabs.forEach((button) => {
  button.addEventListener("click", () => {
    const target = button.dataset.table;

    tabs.forEach((tab) => tab.classList.toggle("active", tab === button));
    tableViews.forEach((view) => {
      view.classList.toggle("active", view.id === `table-${target}`);
    });
  });
});

const metricObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;

      const value = entry.target.dataset.count;
      if (!value || entry.target.dataset.done === "true") return;

      entry.target.dataset.done = "true";
      const target = Number(value);
      const start = performance.now();
      const duration = 850;

      function tick(now) {
        const ratio = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - ratio, 3);
        entry.target.textContent = (target * eased).toFixed(4);
        if (ratio < 1) requestAnimationFrame(tick);
      }

      requestAnimationFrame(tick);
    });
  },
  { threshold: 0.35 }
);

document.querySelectorAll("[data-count]").forEach((item) => metricObserver.observe(item));

const barObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.querySelectorAll(".bar i").forEach((bar) => {
        const width = bar.style.width;
        bar.style.width = "0";
        requestAnimationFrame(() => {
          bar.style.width = width;
        });
      });
      barObserver.unobserve(entry.target);
    });
  },
  { threshold: 0.25 }
);

const chart = document.querySelector(".bar-chart");
if (chart) barObserver.observe(chart);
