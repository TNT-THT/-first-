// 导出模型时选择保存位置：浏览器能力检测、选择框参数、取消识别
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isPickerCancelled, savePickerOptions, supportsSaveLocationPicker } from '../web/src/save-location.js';
import { modelFileName } from '../web/src/model-file.js';

test('supportsSaveLocationPicker：只有提供 showSaveFilePicker 的浏览器才支持', () => {
  assert.equal(supportsSaveLocationPicker({ showSaveFilePicker() {} }), true);
  assert.equal(supportsSaveLocationPicker({}), false);
  assert.equal(supportsSaveLocationPicker(undefined), false);
});

test('savePickerOptions：默认文件名与 .ojmodel 过滤器', () => {
  const o = savePickerOptions('Qwen2.5-3B.ojmodel');
  assert.equal(o.suggestedName, 'Qwen2.5-3B.ojmodel');
  assert.deepEqual(o.types[0].accept, { 'application/octet-stream': ['.ojmodel'] });
});

test('isPickerCancelled：用户取消选择框（AbortError）时返回 true，其他错误返回 false', () => {
  assert.equal(isPickerCancelled(Object.assign(new Error('x'), { name: 'AbortError' })), true);
  assert.equal(isPickerCancelled(new Error('磁盘已满')), false);
  assert.equal(isPickerCancelled(undefined), false);
});

test('modelFileName：模型 id 中的不安全字符被替换，并带 .ojmodel 扩展名', () => {
  assert.equal(modelFileName('Qwen2.5-3B-Instruct-q4f16_1-MLC'), 'Qwen2.5-3B-Instruct-q4f16_1-MLC.ojmodel');
  assert.equal(modelFileName('a b/c'), 'a_b_c.ojmodel');
});
