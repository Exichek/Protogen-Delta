// Проверка настоящего интерфейса, XHR, плееров и скачивания без Telegram и модели.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const {execFileSync}=require('node:child_process');
const {chromium}=require('playwright');
const root=path.join(__dirname,'../src/protogen_delta/miniapp/static');
const python=process.env.MINIAPP_PYTHON || path.join(__dirname,'../.venv/Scripts/python.exe');
const headers=JSON.parse(execFileSync(python,['-c','import asyncio,json; from protogen_delta.miniapp.server import MiniAppServer; from protogen_delta.core.user_state import UserStateStore; r=asyncio.run(MiniAppServer("synthetic",UserStateStore())._index(None)); print(json.dumps(dict(r.headers)))'],{encoding:'utf8',env:{...process.env,PYTHONUTF8:'1'}}));
const profile={user:{id:42,first_name:'Тест'},content_mode:'soft',roleplay_active:false,roleplay_configuration:'male',roleplay_character:'PRIVATE',roleplay_preferences:'PRIVATE',roleplay_boundaries:'',roleplay_fetishes:[],delta_appearance:'',delta_species:null,delta_appearance_topics:[],appearance_upload_enabled:true,delta_appearance_profile:{personality:'',behavior:'',preferences:'',fetishes:'',boundaries:''}};
async function main(){
  let checks=0,failed=false,requests=[],deletions=0,lastFormat='mp4';
  const server=http.createServer((req,res)=>{
    if(req.url==='/'){res.writeHead(200,headers);res.end(fs.readFileSync(path.join(root,'index.html')));return;}
    if(req.url==='/static/app.js'){res.writeHead(200,{'Content-Type':'application/javascript'});res.end(fs.readFileSync(path.join(root,'app.js')));return;}
    if(req.url==='/api/profile'){res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify(profile));return;}
    if(req.method==='DELETE' && req.url.startsWith('/api/tools/convert/')){deletions++;res.writeHead(200,{'Content-Type':'application/json'});res.end('{"deleted":true}');return;}
    if(req.url==='/api/tools/convert/file'){
      assert.equal(req.headers['x-telegram-init-data'],'synthetic');
      const options=JSON.parse(decodeURIComponent(req.headers['x-conversion-options']));
      lastFormat=options.format;
      const chunks=[];req.on('data',chunk=>chunks.push(chunk));req.on('end',()=>{
        const input=Buffer.concat(chunks);requests.push({options,size:input.length});assert(!input.toString().includes('PRIVATE'));
        setTimeout(()=>{if(failed){res.writeHead(400,{'Content-Type':'text/plain'});res.end('Неверный файл');return;}
          res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({token:'synthetic-token',name:'пример.'+options.format,format:options.format,mode:options.mode,bytes:4096,duration:1,expires_in:900,download_url:'/converted-media/synthetic-token/download',preview_url:'/converted-media/synthetic-token/preview?mode='+options.mode}));},100);
      });return;
    }
    if(req.url.startsWith('/converted-media/')){
      const audio=req.url.includes('mode=audio');
      const data=fs.readFileSync(path.join(__dirname,'../.tmp/converter-browser.'+(audio?'mp3':'mp4')));
      const range=/bytes=(\d+)-(\d*)/.exec(req.headers.range || '');
      if(req.url.includes('/download')) res.setHeader('Content-Disposition',"attachment; filename*=UTF-8''"+encodeURIComponent('пример.'+lastFormat));
      res.setHeader('Content-Type',audio?'audio/mpeg':'video/mp4');
      if(range){const start=Number(range[1]),end=range[2]?Math.min(Number(range[2]),data.length-1):data.length-1;res.writeHead(206,{'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${data.length}`,'Content-Length':end-start+1});res.end(data.subarray(start,end+1));}
      else{res.writeHead(200,{'Content-Length':data.length,'Accept-Ranges':'bytes'});res.end(data);}return;
    }
    res.writeHead(404);res.end();
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const origin=`http://127.0.0.1:${server.address().port}`;
  const browser=await chromium.launch({headless:true,channel:process.env.MINIAPP_BROWSER_CHANNEL || undefined});
  const errors=[];
  try{
    const context=await browser.newContext({acceptDownloads:true,viewport:{width:430,height:800}});
    await context.route('**/*',route=>{const url=route.request().url();if(url.startsWith(origin))return route.continue();if(url.startsWith('https://telegram.org/js/telegram-web-app.js'))return route.fulfill({contentType:'application/javascript',body:'window.Telegram={WebApp:{initData:"synthetic",ready(){},expand(){}}}'});return route.abort();});
    const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));
    await page.goto(origin);await page.waitForFunction(()=>document.querySelector('#hello').textContent.includes('42'));
    await page.locator('#tab-tools').click();
    assert(await page.locator('#conversion-submit').isDisabled());
    assert.deepEqual(await page.locator('#conversion-format option').evaluateAll(nodes=>nodes.map(n=>n.value)),['mp4','avi','mov','mkv']);checks++;
    await page.locator('#conversion-file').setInputFiles({name:'пример.mp4',mimeType:'video/mp4',buffer:fs.readFileSync(path.join(__dirname,'../.tmp/converter-browser.mp4'))});
    assert(await page.locator('#conversion-submit').isEnabled());checks++;
    await page.locator('#conversion-format').selectOption('mkv');
    await page.locator('#conversion-submit').click();
    assert(await page.locator('#conversion-format').isDisabled());checks++;
    await page.waitForFunction(()=>!document.querySelector('#conversion-result').hidden);
    await page.waitForFunction(()=>document.querySelector('#conversion-video').readyState>=2);
    assert.equal(requests[0].options.format,'mkv');assert.equal(requests[0].options.mode,'video');
    assert(await page.locator('#conversion-preview-hint').textContent().then(text=>text.includes('отдельное превью')));checks++;
    await page.locator('#conversion-video').evaluate(async element=>{element.muted=true;await element.play();});
    await page.waitForFunction(()=>document.querySelector('#conversion-video').currentTime>0);checks++;
    const downloadPromise=page.waitForEvent('download');await page.locator('#conversion-save').click();const download=await downloadPromise;
    assert.equal(download.suggestedFilename(),'пример.mkv');await download.saveAs(path.join(__dirname,'../.tmp/converter-browser-saved.mkv'));checks++;
    const screenshots=path.join(__dirname,'../.tmp/converter-preview');fs.mkdirSync(screenshots,{recursive:true});
    for(const width of [320,430,720,1280]){
      await page.setViewportSize({width,height:800});
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      const box=await page.locator('#conversion-video').boundingBox();assert(box.width<=width-32);checks++;
      if(width===430 || width===1280) {await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(screenshots,width===430?'mobile.png':'desktop.png'),fullPage:true});}
    }
    await page.locator('[data-conversion-mode="audio"]').click();
    assert.deepEqual(await page.locator('#conversion-format option').evaluateAll(nodes=>nodes.map(n=>n.value)),['mp3','wav','flac']);
    assert(await page.locator('#conversion-resolution-field').isHidden());assert(await page.locator('#conversion-bitrate-field').isVisible());checks++;
    await page.locator('#conversion-format').selectOption('flac');assert(await page.locator('#conversion-bitrate-field').isHidden());checks++;
    await page.locator('#conversion-submit').click();await page.waitForFunction(()=>!document.querySelector('#conversion-result').hidden);
    await page.waitForFunction(()=>document.querySelector('#conversion-audio').readyState>=2);
    assert(await page.locator('#conversion-video').isHidden());assert(await page.locator('#conversion-audio').isVisible());checks++;
    await page.locator('#conversion-clear').click();assert(await page.locator('#conversion-result').isHidden());assert(await page.locator('#conversion-submit').isDisabled());checks++;
    await page.locator('#conversion-file').setInputFiles({name:'empty.wav',mimeType:'audio/wav',buffer:Buffer.alloc(0)});
    assert(await page.locator('#conversion-status').textContent().then(text=>text.includes('непустой')));checks++;
    await page.locator('#conversion-file').setInputFiles({name:'bad.mp4',mimeType:'video/mp4',buffer:Buffer.from('synthetic')});failed=true;
    await page.locator('#conversion-submit').click();await page.waitForFunction(()=>document.querySelector('#conversion-status').classList.contains('error'));
    assert(await page.locator('#conversion-submit').isEnabled());assert(await page.locator('#conversion-result').isHidden());checks++;
    assert(deletions>=2);assert.equal(errors.length,0,errors.join('\n'));
    console.log(JSON.stringify({checks,real_xhr:true,video_playback:true,audio_preview:true,download:true,responsive:true,production_csp:true,page_errors:0,external_requests:0}));
  }finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
