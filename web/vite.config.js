import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root,
  base: './',
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
