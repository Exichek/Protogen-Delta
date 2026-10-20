// Проверка поиска пользователя в настоящем интерфейсе без внешних запросов.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const {execFileSync}=require('node:child_process');
const {chromium}=require('playwright');
const root=path.join(__dirname,'../src/protogen_delta/miniapp/static');
const python=process.env.MINIAPP_PYTHON || path.join(__dirname,'../.venv/Scripts/python.exe');
const headers=JSON.parse(execFileSync(python,['-c','import asyncio,json; from protogen_delta.miniapp.server import MiniAppServer; from protogen_delta.core.user_state import UserStateStore; r=asyncio.run(MiniAppServer("synthetic",UserStateStore())._index(None)); print(json.dumps(dict(r.headers)))'],{encoding:'utf8',env:{...process.env,PYTHONUTF8:'1'}}));
const profile={user:{id:42,first_name:'Тест'},content_mode:'soft',roleplay_active:false,roleplay_configuration:'male',roleplay_character:'',roleplay_preferences:'',roleplay_boundaries:'',roleplay_fetishes:[],delta_appearance:'',delta_species:null,delta_appearance_topics:[],appearance_upload_enabled:true,delta_appearance_profile:{personality:'',behavior:'',preferences:'',fetishes:'',boundaries:''}};
async function main(){
  let mode='profile', requests=[], checks=0;
  const server=http.createServer((req,res)=>{
    if(req.url==='/'){res.writeHead(200,headers);res.end(fs.readFileSync(path.join(root,'index.html')));return;}
    if(['/static/app.js','/static/markets.js'].includes(req.url)){res.writeHead(200,{'Content-Type':'application/javascript'});res.end(fs.readFileSync(path.join(root,path.basename(req.url))));return;}
    if(req.url==='/api/profile'){res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify(profile));return;}
    if(req.url==='/api/tools/id' && req.method==='POST'){
      assert.equal(req.headers['x-telegram-init-data'],'synthetic');
      const chunks=[];req.on('data',part=>chunks.push(part));req.on('end',()=>{
        const payload=JSON.parse(Buffer.concat(chunks).toString());requests.push(payload);
        setTimeout(()=>{
          if(mode==='error'){res.writeHead(400,{'Content-Type':'text/plain;charset=utf-8'});res.end('Telegram не дал профиль по этому ID.');return;}
          let result={id:9876543210,title:'Тест Профиль',username:'synthetic_user',type:'private',message:'Профиль Telegram: Тест Профиль\nID: 9876543210\n@synthetic_user'};
          if(mode==='no-username') result={...result,username:null,message:'Тест без username\nID: 9876543210'};
          if(mode==='unsafe') result={...result,username:'evil/path',message:'<img src=x onerror="window.injected=true">'};
          if(payload.target==='self') result={id:42,title:'Твой Telegram ID',type:'private'};
          if(payload.target==='@public_group') result={id:-100,title:'Публичная группа',type:'supergroup'};
          if(payload.target==='select') result={message:'Выбор ID открыт в чате с Дельтой.'};
          res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(result));
        },100);
      });return;
    }
    res.writeHead(404);res.end();
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const origin=`http://127.0.0.1:${server.address().port}`;
  const browser=await chromium.launch({headless:true,channel:process.env.MINIAPP_BROWSER_CHANNEL || undefined});
  const errors=[];
  try{
    const context=await browser.newContext();
    await context.route('**/*',route=>{
      const url=route.request().url();
      if(url.startsWith(origin)) return route.continue();
      if(url.startsWith('https://telegram.org/js/telegram-web-app.js')) return route.fulfill({contentType:'application/javascript',body:'window.openedProfiles=[];window.Telegram={WebApp:{initData:"synthetic",ready(){},expand(){},openTelegramLink(url){window.openedProfiles.push(url)}}}'});
      return route.abort();
    });
    for(const viewport of [{width:390,height:844},{width:1280,height:720}]){
      mode='profile';const page=await context.newPage();await page.setViewportSize(viewport);
      page.on('pageerror',error=>errors.push(error.message));
      await page.goto(origin);await page.waitForFunction(()=>document.querySelector('#hello').textContent.includes('42'));
      await page.locator('#tab-tools').click();
      const input=page.locator('#id-target'), lookup=page.locator('#lookup-id'), output=page.locator('#id-status'), link=page.locator('#id-profile-link');
      await input.fill('9876543210');await input.press('Enter');
      assert(await page.locator('#my-id').isDisabled());
      await page.waitForFunction(()=>document.querySelector('#id-status').textContent.includes('@synthetic_user'));
      assert.equal(requests.at(-1).target,'9876543210');assert.equal(await link.getAttribute('href'),'https://t.me/synthetic_user');
      await link.click();assert.deepEqual(await page.evaluate(()=>window.openedProfiles),['https://t.me/synthetic_user']);checks++;
      await page.evaluate(()=>Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:text=>new Promise(resolve=>{window.copiedValue=text;window.finishCopy=resolve;})}}));
      await page.locator('#copy-id').click();assert.equal(await page.evaluate(()=>window.copiedValue),'9876543210');
      mode='error';await lookup.click();await page.waitForFunction(()=>document.querySelector('#id-status').classList.contains('error'));
      assert(await link.isHidden());assert.equal(await link.getAttribute('href'),null);assert(await page.locator('#copy-id').isHidden());checks++;
      await page.evaluate(()=>window.finishCopy());assert((await output.textContent()).includes('Telegram не дал профиль'));checks++;
      mode='no-username';await lookup.click();await page.waitForFunction(()=>document.querySelector('#id-status').textContent.includes('без username'));
      assert(await link.isHidden());assert(await page.locator('#copy-id').isVisible());checks++;
      mode='unsafe';await lookup.click();await page.waitForFunction(()=>document.querySelector('#id-status').textContent.startsWith('<img'));
      assert.equal(await output.locator('img').count(),0);assert(await page.evaluate(()=>!window.injected));assert(await link.isHidden());checks++;
      mode='profile';await input.fill('@public_group');await lookup.click();await page.waitForFunction(()=>document.querySelector('#id-status').textContent.includes('Публичная группа'));
      assert((await output.textContent()).includes('ID: -100'));assert(await link.isHidden());checks++;
      await page.locator('#my-id').click();await page.waitForFunction(()=>document.querySelector('#id-status').textContent.includes('ID: 42'));checks++;
      await page.locator('#select-id').click();await page.waitForFunction(()=>document.querySelector('#id-status').textContent.includes('Выбор ID открыт'));
      assert(await page.locator('#copy-id').isHidden());checks++;
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
      await page.close();
    }
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({checks,viewports:2,page_errors:0,external_requests:0}));
  } finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
