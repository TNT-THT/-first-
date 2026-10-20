// WebLLM 推理放在 Web Worker 中运行，避免模型计算时页面卡顿。
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm';

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg) => handler.onmessage(msg);
