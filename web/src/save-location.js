/**
 * 导出时选择保存位置（纯逻辑，便于测试）。
 *
 * 支持 File System Access API（showSaveFilePicker）的浏览器（Chrome、Edge 等桌面版）可以让用户
 * 自己选择保存的盘符与文件夹。其他浏览器（如 Firefox、Safari）没有这个接口，只能使用浏览器的下载目录。
 */

export const MODEL_FILE_PICKER_TYPES = [
  {
    description: 'OJ 模型文件（.ojmodel）',
    accept: { 'application/octet-stream': ['.ojmodel'] },
  },
];

/** 当前浏览器是否支持选择保存位置。 */
export function supportsSaveLocationPicker(win = globalThis) {
  return typeof win?.showSaveFilePicker === 'function';
}

/** showSaveFilePicker 的参数。 */
export function savePickerOptions(suggestedName) {
  return { suggestedName, types: MODEL_FILE_PICKER_TYPES };
}

/** 用户在选择框中点了“取消”（浏览器以 AbortError 结束）。 */
export function isPickerCancelled(err) {
  return err?.name === 'AbortError';
}

/** 生图模型文件（.ojimage）的保存类型。 */
export const IMAGE_FILE_PICKER_TYPES = [
  {
    description: 'OJ 生图模型文件（.ojimage）',
    accept: { 'application/octet-stream': ['.ojimage'] },
  },
];

/** 生图模型导出时 showSaveFilePicker 的参数。 */
export function imagePickerOptions(suggestedName) {
  return { suggestedName, types: IMAGE_FILE_PICKER_TYPES };
}
