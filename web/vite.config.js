import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const root = path.dirname(fileURLToPath(import.meta.url));

// 生图用的 ONNX Runtime wasm 运行库（与 package.json 中 onnxruntime-web 版本一致）。
// 构建时输出到 dist/ort/，随网站包一起分发，因此不依赖 CDN；开发服务器从 node_modules 直接提供。
const ORT_DIST = path.join(root, '..', 'node_modules', 'onnxruntime-web', 'dist');
// onnxruntime-web/webgpu 的默认入口（bundle）通过 wasmPaths 加载 asyncify 变体，这两个文件必须随网站包提供。
const ORT_FILES = ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];
const ORT_TYPES = { '.mjs': 'text/javascript', '.wasm': 'application/wasm' };

function ortRuntime() {
  return {
    name: 'oj-ort-runtime',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const m = /^\/ort\/([^/?]+)/.exec(req.url || '');
        if (!m || !ORT_FILES.includes(m[1])) return next();
        const file = path.join(ORT_DIST, m[1]);
        res.setHeader('Content-Type', ORT_TYPES[path.extname(file)] || 'application/octet-stream');
        fs.createReadStream(file)
          .on('error', () => {
            res.statusCode = 404;
            res.end();
          })
          .pipe(res);
      });
    },
    generateBundle() {
      // 远程引擎构建不包含本机生图，无需输出运行库
      if (process.env.VITE_ENGINE === 'remote') return;
      for (const f of ORT_FILES) {
        this.emitFile({ type: 'asset', fileName: `ort/${f}`, source: fs.readFileSync(path.join(ORT_DIST, f)) });
      }
    },
  };
}

export default defineConfig({
  root,
  base: './',
  plugins: [ortRuntime()],
  build: {
    outDir: path.join(root, 'dist'),
    emptyOutDir: true,
    target: 'es2020',
    assetsInlineLimit: 0,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    // 允许沙箱预览域名访问开发服务（仅影响 dev server）
    allowedHosts: ['.e2b.app'],
    // 开发模式下把 /api 转发给后端，保证浏览器端只使用相对路径
    proxy: {
      '/api': `http://127.0.0.1:${process.env.PORT || 3000}`,
    },
  },
});
