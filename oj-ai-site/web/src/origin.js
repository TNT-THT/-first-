/**
 * 判断父页面 origin 是否在白名单内。
 * 支持三种写法：'*'（任意来源）、精确 origin（https://hydro.ac）、
 * 通配子域名（https://*.hydro.ac，同时匹配 https://a.hydro.ac，不匹配 hydro.ac 本身）。
 */
export function isAllowedOrigin(origin, patterns) {
  if (!origin || origin === 'null') return false;
  let url;
  try {
    url = new URL(origin);
  } catch {
    return false;
  }
  return patterns.some((p) => {
    if (p === '*') return true;
    const m = p.match(/^(https?):\/\/\*\.(.+)$/);
    if (m) {
      return url.protocol === `${m[1]}:` && url.hostname.endsWith(`.${m[2]}`);
    }
    return p === url.origin;
  });
}
