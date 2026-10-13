// Встраиваем JS из пакета только в автономные браузерные тесты с setContent.
// CSP проверяется отдельно через HTTP с настоящими заголовками приложения.
const fs = require('node:fs');
const path = require('node:path');
exports.readMiniAppHtml = () => {
  const root = path.join(__dirname, '../src/protogen_delta/miniapp/static');
  return fs.readFileSync(path.join(root, 'index.html'), 'utf8').replace(
    '<script src="/static/app.js" defer></script>',
    () => `<script>${fs.readFileSync(path.join(root, 'app.js'), 'utf8')}</script>`
  );
};
