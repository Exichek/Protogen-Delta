// Real HTTP + production CSP; synthetic Telegram SDK/profile, no external calls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {execFileSync} = require('node:child_process');
const {chromium} = require('playwright');
const root = path.join(__dirname, '../src/protogen_delta/miniapp/static');
const python = process.env.MINIAPP_PYTHON || path.join(__dirname, '../.venv/Scripts/python.exe');
const headers = JSON.parse(execFileSync(python, ['-c',
  'import asyncio,json; from protogen_delta.miniapp.server import MiniAppServer; from protogen_delta.core.user_state import UserStateStore; r=asyncio.run(MiniAppServer("synthetic",UserStateStore())._index(None)); print(json.dumps(dict(r.headers)))'
], {encoding:'utf8', env:{...process.env,PYTHONUTF8:'1'}}));
const profile = {user:{id:42,first_name:'Тест'},content_mode:'soft',roleplay_active:false,roleplay_configuration:'male',roleplay_character:'',roleplay_preferences:'',roleplay_boundaries:'',roleplay_fetishes:[],delta_appearance:'',delta_species:null,delta_appearance_topics:[],appearance_upload_enabled:true,delta_appearance_profile:{personality:'',behavior:'',preferences:'',fetishes:'',boundaries:''}};
async function main() {
  const server = http.createServer((request,response) => {
    if(request.url === '/') {
      response.writeHead(200,headers);response.end(fs.readFileSync(path.join(root,'index.html')));
    } else if(request.url === '/static/app.js') {
      response.writeHead(200,{'Content-Type':'application/javascript','X-Content-Type-Options':'nosniff'});response.end(fs.readFileSync(path.join(root,'app.js')));
    } else if(request.url === '/api/profile') {
      assert.equal(request.headers['x-telegram-init-data'],'synthetic');
      response.writeHead(200,{'Content-Type':'application/json'});response.end(JSON.stringify(profile));
    } else {
      response.writeHead(404);response.end();
    }
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({headless:true,channel:process.env.MINIAPP_BROWSER_CHANNEL || undefined});
  try {
    const context = await browser.newContext({viewport:{width:430,height:800}});
    let sdkRequests=0;const errors=[];
    await context.route('**/*', async route => {
      if(route.request().url().startsWith(origin)) return route.continue();
      if(route.request().url().startsWith('https://telegram.org/js/telegram-web-app.js')) {
        sdkRequests++;return route.fulfill({contentType:'application/javascript',body:'window.Telegram={WebApp:{initData:"synthetic",ready(){},expand(){}}}'});
      }
      return route.abort();
    });
    const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));
    const response=await page.goto(origin);
    assert(!response.headers()['content-security-policy'].split(';').find(rule=>rule.trim().startsWith('script-src')).includes('unsafe-inline'));
    await page.waitForFunction(()=>document.querySelector('#hello').textContent.includes('42'));
    assert.equal(sdkRequests,1);
    await page.locator('#tab-translate').click();
    await page.locator('#translation-input').fill('Hello');
    assert(await page.locator('#translation-submit').isEnabled());
    await page.evaluate(()=>{
      window.cspViolations=[];document.addEventListener('securitypolicyviolation',event=>window.cspViolations.push(event.violatedDirective));
      const script=document.createElement('script');script.textContent='window.inlineAttack=true';document.body.append(script);
      const image=document.createElement('img');image.setAttribute('onerror','window.handlerAttack=true');image.src='/missing-image';document.body.append(image);
    });
    await page.waitForFunction(()=>window.cspViolations.length>=2);
    assert(await page.evaluate(()=>!window.inlineAttack && !window.handlerAttack));
    assert.equal(errors.length,0,errors.join('\n'));
    await browser.close();
    console.log(JSON.stringify({production_csp:true,external_application_js_runs:true,telegram_sdk_allowed:true,inline_script_blocked:true,inline_handler_blocked:true,page_errors:0,external_requests:0}));
  } finally {
    await browser.close();await new Promise(resolve=>server.close(resolve));
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
