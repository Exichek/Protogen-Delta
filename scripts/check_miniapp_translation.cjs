// Real browser + synthetic API, files and clipboard. No paid model/Telegram calls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const html=fs.readFileSync(path.join(__dirname,'../src/protogen_delta/miniapp/static/index.html'),'utf8');
const profile={user:{first_name:'Тест',id:42},content_mode:'soft',roleplay_active:true,roleplay_configuration:'male',roleplay_character:'PRIVATE CHARACTER',roleplay_preferences:'PRIVATE PREFS',roleplay_boundaries:'',roleplay_fetishes:[],appearance_upload_enabled:true,delta_appearance:'PRIVATE APPEARANCE',delta_species:null,delta_appearance_profile:{personality:'',behavior:'',preferences:'',fetishes:'',boundaries:''},delta_appearance_topics:[]};

async function main() {
  const browser=await chromium.launch({headless:true,ignoreDefaultArgs:['--hide-scrollbars'],channel:process.env.MINIAPP_BROWSER_CHANNEL || undefined});
  const context=await browser.newContext({viewport:{width:430,height:800},acceptDownloads:true});
  let checks=0; const errors=[];
  await context.route('**/*',route=>route.abort());
  try {
    const page=await context.newPage(); page.on('pageerror',error=>errors.push(error.message));
    const setup=`<script>
      window.Telegram={WebApp:{initData:'synthetic',ready(){},expand(){}}};
      window.profile=${JSON.stringify(profile)};window.calls=[];window.fail=false;window.deferred=false;window.copied='';
      Object.defineProperty(navigator,'clipboard',{value:{writeText:async(text)=>{window.copied=text;},read:async()=>[{types:['image/png'],getType:async()=>new Blob(['synthetic'],{type:'image/png'})}]}});
      window.fetch=async(url,options)=>{
        const body=options.body instanceof File ? {file:options.body.name} : options.body ? JSON.parse(options.body) : null;
        window.calls.push({url,body,headers:options.headers});
        if(url==='/api/profile')return {ok:true,json:async()=>structuredClone(window.profile)};
        if(window.fail)return {ok:false,status:502,text:async()=>'Не удалось получить перевод'};
        if(window.deferred)await new Promise(resolve=>{window.resolveRequest=resolve;});
        if(url==='/api/tools/translate/export')return {ok:true,blob:async()=>new Blob(['PK-synthetic-docx'],{type:'application/vnd.openxmlformats-officedocument.wordprocessingml.document'})};
        return {ok:true,json:async()=>({text:window.resultText || 'Привет, мир!',source_text:'Hello, world!',detected_source:'en',target:'ru'})};
      };
    </script>`;
    await page.setContent(html.replace(/<script src="https:\/\/telegram.org[^>]+><\/script>/,setup));
    await page.waitForFunction(()=>document.querySelector('#hello').textContent.includes('42'));
    await page.locator('#tab-translate').click();
    assert(await page.locator('#translate-panel').isVisible());
    assert(await page.locator('#rp-panel').isHidden());
    assert(await page.locator('#translation-submit').isDisabled()); checks++;
    await page.locator('#translation-input').fill('Hello, world!');
    await page.locator('#translation-submit').click();
    await page.waitForFunction(()=>document.querySelector('#translation-output').value==='Привет, мир!');
    assert.equal(await page.locator('#translation-detected').textContent(),'Определён язык: Английский');
    let request=await page.evaluate(()=>window.calls.find(call=>call.url==='/api/tools/translate'));
    assert.deepEqual(request.body,{text:'Hello, world!',source:'auto',target:'ru'});
    assert.equal(request.headers['X-Telegram-Init-Data'],'synthetic');
    assert(!JSON.stringify(request).includes('PRIVATE')); checks++;
    await page.locator('#translation-copy').click();
    assert.equal(await page.evaluate(()=>window.copied),'Привет, мир!'); checks++;
    await page.locator('#translation-swap').click();
    assert.equal(await page.locator('#translation-source').inputValue(),'ru');
    assert.equal(await page.locator('#translation-target').inputValue(),'en');
    assert.equal(await page.locator('#translation-input').inputValue(),'Привет, мир!');
    assert.equal(await page.locator('#translation-output').inputValue(),''); checks++;

    await page.evaluate(()=>{window.fail=true;});
    await page.locator('#translation-submit').click();
    await page.waitForFunction(()=>document.querySelector('#translation-status').textContent==='Не удалось получить перевод');
    assert.equal(await page.locator('#translation-input').inputValue(),'Привет, мир!');
    assert(await page.locator('#translation-submit').isEnabled()); checks++;
    await page.evaluate(()=>{window.fail=false;window.deferred=true;});
    await page.locator('#translation-submit').click();
    await page.waitForFunction(()=>Boolean(window.resolveRequest));
    await page.locator('#translation-clear').click();
    await page.evaluate(()=>{window.deferred=false;window.resolveRequest();});
    await page.waitForFunction(()=>!translationBusy);
    assert.equal(await page.locator('#translation-output').inputValue(),''); checks++;

    await page.locator('[data-translation-mode="image"]').click();
    await page.locator('#translation-file').setInputFiles({name:'photo.png',mimeType:'image/png',buffer:Buffer.from('synthetic')});
    assert(await page.locator('#translation-preview').isVisible());
    await page.locator('#translation-submit').click();
    await page.waitForFunction(()=>document.querySelector('#translation-output').value==='Привет, мир!');
    request=await page.evaluate(()=>window.calls.filter(call=>call.url==='/api/tools/translate/file').at(-1));
    assert.deepEqual(JSON.parse(decodeURIComponent(request.headers['X-Translation-Options'])),{mode:'image',source:'ru',target:'en',file_name:'photo.png'});
    assert.equal(await page.locator('#translation-input').inputValue(),'Hello, world!');
    assert.equal(await page.locator('#character').inputValue(),'PRIVATE CHARACTER'); checks++;
    await page.locator('#translation-paste').click();
    await page.waitForFunction(()=>document.querySelector('#translation-file-name').textContent.includes('clipboard.png'));
    const appearanceBefore=await page.locator('#appearance').textContent();
    await page.locator('#translation-upload').evaluate(node=>{
      const data=new DataTransfer();data.items.add(new File(['paste'],'paste.png',{type:'image/png'}));
      node.dispatchEvent(new ClipboardEvent('paste',{clipboardData:data,bubbles:true,cancelable:true}));
    });
    assert((await page.locator('#translation-file-name').textContent()).includes('paste.png'));
    assert.equal(await page.locator('#appearance').textContent(),appearanceBefore); checks++;
    await page.locator('#translation-upload').evaluate(node=>{
      const data=new DataTransfer();data.items.add(new File(['drop'],'drop.webp',{type:'image/webp'}));
      node.dispatchEvent(new DragEvent('drop',{dataTransfer:data,bubbles:true,cancelable:true}));
    });
    assert((await page.locator('#translation-file-name').textContent()).includes('drop.webp')); checks++;

    await page.locator('[data-translation-mode="document"]').click();
    await page.locator('#translation-file').setInputFiles({name:'note.txt',mimeType:'text/plain',buffer:Buffer.from('Hello')});
    await page.locator('#translation-submit').click();
    await page.waitForFunction(()=>document.querySelector('#translation-output').value==='Привет, мир!');
    request=await page.evaluate(()=>window.calls.filter(call=>call.url==='/api/tools/translate/file').at(-1));
    assert.equal(JSON.parse(decodeURIComponent(request.headers['X-Translation-Options'])).mode,'document');
    const txtPromise=page.waitForEvent('download');await page.locator('#translation-txt').click();
    assert.equal((await txtPromise).suggestedFilename(),'translation-en.txt');
    const docxPromise=page.waitForEvent('download');await page.locator('#translation-docx').click();
    assert.equal((await docxPromise).suggestedFilename(),'translation-en.docx'); checks++;

    const longText='Очень длинный текст перевода с переносами строк.\n'.repeat(1500);
    await page.evaluate(text=>{document.querySelector('#translation-input').value=text;document.querySelector('#translation-output').value=text;},longText);
    fs.mkdirSync(path.join(__dirname,'../.tmp/translator-preview'),{recursive:true});
    for(const [width,height] of [[320,600],[360,740],[430,800],[650,500],[1024,768],[1440,900]]) {
      await page.setViewportSize({width,height});
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      for(const id of ['translation-input','translation-output']) {
        const box=page.locator('#'+id); const bounds=await box.boundingBox();
        assert(bounds.height<=331);
        assert(await box.evaluate(node=>node.scrollHeight>node.clientHeight));
        await box.evaluate(node=>{node.scrollTop=10000;});
        assert(await box.evaluate(node=>node.scrollTop>0));
      }
      checks++;
    }
    await page.locator('[data-translation-mode="text"]').click();
    await page.evaluate(text=>{document.querySelector('#translation-input').value=text.slice(0,4900);document.querySelector('#translation-output').value=text;updateTranslationButtons();},longText);
    await page.setViewportSize({width:430,height:800});
    await page.screenshot({path:path.join(__dirname,'../.tmp/translator-preview/mobile.png'),fullPage:true});
    await page.setViewportSize({width:1440,height:900});
    await page.evaluate(()=>{
      const style=document.documentElement.style;
      for(const [name,value] of Object.entries({'bg-color':'#ffffff','secondary-bg-color':'#f1f4f9','text-color':'#202124','hint-color':'#68717f','button-color':'#1a73e8','button-text-color':'#ffffff'})) style.setProperty('--tg-theme-'+name,value);
    });
    await page.screenshot({path:path.join(__dirname,'../.tmp/translator-preview/desktop.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    assert.equal(await page.evaluate(()=>window.calls.some(call=>!call.url.startsWith('/api/'))),false);
    console.log(JSON.stringify({checks,page_errors:0,external_requests:0,real_clipboard_calls:0}));
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
