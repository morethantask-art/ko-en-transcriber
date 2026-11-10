// 오프라인 지원: 앱 파일과 CDN 라이브러리(전사 엔진)를 캐시한다.
// 모델 파일은 transformers.js가 자체 캐시('transformers-cache')에 저장한다.
const APP = 'app-v5';
const LIB = 'lib-v1';
const SHELL = ['./', 'index.html', 'worker.js', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png', 'selftest.wav'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(APP).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => ![APP, LIB, 'transformers-cache'].includes(k)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;

  // CDN 라이브러리: 버전이 고정돼 있으니 캐시 우선
  if (url.hostname === 'cdn.jsdelivr.net') {
    e.respondWith(
      caches.open(LIB).then(async (c) => {
        const hit = await c.match(e.request);
        if (hit) return hit;
        const res = await fetch(e.request);
        if (res.ok) c.put(e.request, res.clone());
        return res;
      })
    );
    return;
  }

  // 앱 파일: 네트워크 우선, 실패하면(오프라인) 캐시
  if (url.origin === location.origin) {
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          if (res.ok) caches.open(APP).then((c) => c.put(e.request, res.clone()));
          return isolate(res);
        })
        .catch(() => caches.match(e.request, { ignoreSearch: true }).then((r) => r || caches.match('index.html')).then(isolate))
    );
  }
});

// 전사 엔진(onnxruntime)은 페이지가 'cross-origin isolated'일 때만 CPU 여러 개(스레드)를 쓴다.
// GitHub Pages는 그 헤더를 보낼 수 없어서, 서비스 워커가 앱 파일 응답에 붙여 준다.
// 한 개 → 네 개 스레드로 CPU 전사가 2~3배 빨라진다.
// credentialless: CDN·모델 서버 파일을 쿠키 없이 받는 조건이라 따로 설정하지 않아도 된다.
function isolate(res) {
  if (!res || res.status === 0 || res.type === 'opaque' || res.type === 'opaqueredirect') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Embedder-Policy', 'credentialless');
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}
