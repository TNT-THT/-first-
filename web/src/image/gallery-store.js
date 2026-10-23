/**
 * 生图画廊的本机存储（IndexedDB）。图片以 Blob 保存在用户浏览器中，不上传，刷新页面后仍然保留。
 * 每条记录：{id, blob, filename, prompt, width, height, steps, guidance, seed, model, createdAt}
 * 接口可注入 indexedDB（便于测试）；默认使用 globalThis.indexedDB。
 */

export const GALLERY_DB = 'oj-ai-gallery';
export const GALLERY_STORE = 'images';

let dbPromise = null;

function requestToPromise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('IndexedDB 请求失败'));
  });
}

/** 打开（必要时创建）画廊数据库。同一页面内只打开一次。 */
export function openGallery(factory = globalThis.indexedDB) {
  if (!factory) return Promise.reject(new Error('当前浏览器不支持 IndexedDB，无法保存画廊'));
  if (dbPromise && factory === globalThis.indexedDB) return dbPromise;
  const p = new Promise((resolve, reject) => {
    const req = factory.open(GALLERY_DB, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(GALLERY_STORE)) {
        const store = db.createObjectStore(GALLERY_STORE, { keyPath: 'id' });
        store.createIndex('createdAt', 'createdAt');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('无法打开画廊数据库'));
    req.onblocked = () => reject(new Error('画廊数据库被其他页面占用，请关闭其他标签页后重试'));
  });
  if (factory === globalThis.indexedDB) {
    dbPromise = p;
    p.catch(() => {
      dbPromise = null;
    });
  }
  return p;
}

async function run(mode, fn, factory) {
  const db = await openGallery(factory);
  const tx = db.transaction(GALLERY_STORE, mode);
  const store = tx.objectStore(GALLERY_STORE);
  const result = await fn(store);
  await new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error('画廊写入失败'));
    tx.onabort = () => reject(tx.error || new Error('画廊写入被中止（可能存储空间不足）'));
  });
  return result;
}

/** 按创建时间从新到旧列出全部图片。 */
export async function listGallery(factory) {
  const rows = await run('readonly', (store) => requestToPromise(store.getAll()), factory);
  return rows.sort((a, b) => b.createdAt - a.createdAt);
}

export async function addGalleryImage(record, factory) {
  if (!record?.id || !record.blob) throw new Error('图片记录无效');
  await run('readwrite', (store) => {
    store.put(record);
  }, factory);
  return record;
}

export async function deleteGalleryImage(id, factory) {
  await run('readwrite', (store) => {
    store.delete(id);
  }, factory);
}

export async function clearGallery(factory) {
  await run('readwrite', (store) => {
    store.clear();
  }, factory);
}

/** 测试用：关闭并忘记缓存的数据库连接。 */
export function resetGalleryConnection() {
  dbPromise = null;
}
