self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open('v1').then(function(cache) {
      return cache.addAll([
        '/',                                 // 루트 경로
        '/static/manifest.json',             // PWA 매니페스트
        '/static/service_worker.js'          // 이 파일 자신
        // '/static/style.css' 삭제됨
        // '/index_pwa.html' 삭제됨 (Flask 라우팅에 의해 동작)
      ]);
    })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request).then(function(response) {
      return response || fetch(event.request);
    })
  );
});
