
const videos = Array.from(document.querySelectorAll('video'));
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const autoPaused = new WeakSet();
const manuallyPaused = new WeakSet();
for (const video of videos) {
  video.addEventListener('pause', () => {
    if (autoPaused.has(video)) autoPaused.delete(video);
    else manuallyPaused.add(video);
  });
  video.addEventListener('play', () => manuallyPaused.delete(video));
}
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      const video = entry.target;
      if (entry.isIntersecting && !reducedMotion && !manuallyPaused.has(video)) {
        video.play().catch(() => {});
      } else if (!entry.isIntersecting && !video.paused) {
        autoPaused.add(video);
        video.pause();
      }
    }
  }, {threshold: 0.35});
  videos.forEach(video => observer.observe(video));
}
