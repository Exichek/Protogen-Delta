// Browser regression checks with synthetic clipboard, Telegram and HTTP responses.
// Run with Playwright installed: node scripts/check_miniapp_clipboard.cjs
// Optional: MINIAPP_BROWSER_CHANNEL=msedge; MINIAPP_SCREENSHOTS=/path/to/output
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const html = fs.readFileSync(path.join(__dirname, '../src/protogen_delta/miniapp/static/index.html'), 'utf8');
const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1sAAAAASUVORK5CYII=';
const profile = {
  user: {first_name: 'Тест', id: 1}, content_mode: 'soft', roleplay_configuration: 'male',
  roleplay_character: '', roleplay_preferences: '', roleplay_boundaries: '',
  roleplay_fetishes: [], roleplay_active: false, appearance_upload_enabled: true,
  delta_appearance: '', delta_species: null,
};

async function main() {
  const browser = await chromium.launch({headless: true, channel: process.env.MINIAPP_BROWSER_CHANNEL || undefined});
  let checks = 0;
  const errors = [];
  const context = await browser.newContext({viewport: {width: 430, height: 932}});
  // Block every external request; these tests never access an actual clipboard or bot.
  await context.route('**/*', route => route.abort());
  async function fresh(saved = false, loadState = 'ready', ageRestricted = false) {
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    const setup = `<script>
      window.Telegram={WebApp:{initData:'synthetic',ready(){},expand(){}}};
      window.testProfile=${JSON.stringify(profile)};
      window.testProfile.age_restricted=${ageRestricted};
      const canvas=document.createElement('canvas');canvas.width=32;canvas.height=32;
      const drawing=canvas.getContext('2d');drawing.fillStyle='#38bdd3';drawing.fillRect(0,0,32,32);
      window.thumbnail=canvas.toDataURL('image/jpeg');
      if(${saved})Object.assign(window.testProfile,{delta_appearance:'Сергал',delta_appearance_thumbnail:window.thumbnail});
      window.calls=[]; window.failStatus=0; window.failMessage='';
      window.loadState=${JSON.stringify(loadState)};
      window.fetch=async(url,options)=>{
        window.calls.push({url,method:options.method,headers:options.headers,
          name:options.body?.name,type:options.body?.type,size:options.body?.size});
        if(options.method==='GET' && window.loadState==='pending')await new Promise(resolve=>window.finishProfileLoad=resolve);
        if(options.method==='GET' && window.loadState==='failed'){window.failStatus=401;window.failMessage='initData просрочен';}
        if(!window.failStatus && options.method==='POST' && url==='/api/profile/appearance')
          Object.assign(window.testProfile,{delta_appearance:'Новый облик',delta_appearance_thumbnail:window.thumbnail});
        if(!window.failStatus && options.method==='POST' && url==='/api/profile/character/appearance')
          Object.assign(window.testProfile,{roleplay_character:'Новый персонаж',roleplay_character_thumbnail:window.thumbnail});
        if(!window.failStatus && options.method==='DELETE' && url==='/api/profile/appearance')
          Object.assign(window.testProfile,{delta_appearance:'',delta_appearance_thumbnail:''});
        if(!window.failStatus && options.method==='PATCH' && url==='/api/profile/appearance') {
          window.lastEdit=JSON.parse(options.body);
          window.testProfile.delta_appearance=window.lastEdit.appearance.trim();
        }
        if(!window.failStatus && options.method==='PATCH' && url==='/api/profile') {
          window.lastSettings=JSON.parse(options.body);
          Object.assign(window.testProfile,window.lastSettings);
        }
        return {ok:!window.failStatus,status:window.failStatus||200,
          text:async()=>window.failMessage,json:async()=>window.testProfile};
      };
      window.png='${png}';
      window.makeImage=(type='image/png',name='test.png',size=null)=>new File([
        size===null?Uint8Array.from(atob(window.png),c=>c.charCodeAt(0)):new Uint8Array(size)
      ],name,{type});
      window.setClipboard=(mode='image',type='image/png')=>Object.defineProperty(navigator,'clipboard',{configurable:true,value:{
        read:async()=>{
          if(mode==='denied')throw new DOMException('Blocked','NotAllowedError');
          if(mode==='pending')await new Promise(resolve=>window.finishRead=resolve);
          return mode==='text'?[{types:['text/plain'],getType(){throw Error('Text must not be read');}}]
            :[{types:[type],getType:async()=>window.makeImage(type)}];
        }
      }});
      window.setClipboard();
      window.paste=(target='#appearance-input-area',type='image/png',size=null)=>{
        const data=new DataTransfer(); data.items.add(window.makeImage(type,'pasted.png',size));
        const event=new ClipboardEvent('paste',{clipboardData:data,bubbles:true,cancelable:true});
        document.querySelector(target).dispatchEvent(event); return event.defaultPrevented;
      };
    </script>`;
    await page.setContent(html.replace(/<script src="https:\/\/telegram.org[^>]+><\/script>/, setup));
    if(loadState==='ready')await page.waitForFunction(() => !document.querySelector('#paste-appearance').disabled);
    else if(loadState==='failed')await page.waitForFunction(() => document.querySelector('#appearance-status').className==='error');
    else await page.waitForFunction(() => Boolean(window.finishProfileLoad));
    return page;
  }
  const selected = page => page.locator('#appearance-file-name').textContent();
  const message = page => page.locator('#appearance-status').textContent();
  const waitClipboard = page => page.waitForFunction(() => !document.querySelector('#paste-appearance').disabled);
  try {
    let settings = await fresh();
    await settings.locator('#stopword').fill('Пауза');
    await settings.locator('#save').click();
    await settings.waitForFunction(() => document.querySelector('#status').textContent==='Сохранено.');
    assert.equal(await settings.evaluate(() => window.lastSettings.roleplay_stopword), 'Пауза');
    assert.equal(await settings.locator('#stopword').inputValue(), 'Пауза');
    await settings.close();checks++;
    settings=await fresh(false,'ready',true);
    assert.equal(await settings.locator('#content-mode option[value="adult"]').isDisabled(),true);
    assert.equal(await settings.locator('#age-restriction-hint').isVisible(),true);
    await settings.close();checks++;
    let editor = await fresh(true);
    await editor.locator('#edit-appearance').click();
    assert.equal(await editor.locator('#appearance-edit-text').inputValue(), 'Сергал');
    await editor.locator('#appearance-edit-text').fill('Белая чёлка закрывает один глаз');
    assert.equal(await editor.locator('#apply-appearance').isDisabled(), true);
    if(process.env.MINIAPP_SCREENSHOTS) {
      fs.mkdirSync(process.env.MINIAPP_SCREENSHOTS,{recursive:true});
      await editor.setViewportSize({width:430,height:932});
      await editor.locator('section[aria-labelledby="appearance-title"]').screenshot({path:path.join(process.env.MINIAPP_SCREENSHOTS,'appearance-editor.png')});
    }
    await editor.locator('#cancel-appearance-edit').click();
    assert.equal(await editor.locator('#appearance').textContent(), 'Сергал');
    assert.equal(await editor.evaluate(() => window.calls.length), 1);
    await editor.locator('#edit-appearance').click();
    await editor.locator('#appearance-edit-text').fill('Белая чёлка закрывает один глаз');
    await editor.evaluate(() => {window.failStatus=401;window.failMessage='initData просрочен';});
    await editor.locator('#save-appearance-edit').click();
    await editor.waitForFunction(() => document.querySelector('#appearance-status').className==='error');
    assert.equal(await editor.locator('#appearance-edit-text').inputValue(), 'Белая чёлка закрывает один глаз');
    assert.equal(await editor.locator('#appearance-editor').isVisible(), true);
    await editor.evaluate(() => window.failStatus=0);
    await editor.locator('#save-appearance-edit').click();
    await editor.waitForFunction(() => document.querySelector('#appearance-status').textContent.includes('Описание исправлено'));
    assert.equal(await editor.locator('#appearance').textContent(), 'Белая чёлка закрывает один глаз');
    assert.equal(await editor.locator('#saved-appearance').isVisible(), true);
    assert.equal(await editor.locator('#appearance-editor').isVisible(), false);
    assert.deepEqual(await editor.evaluate(() => window.lastEdit), {appearance:'Белая чёлка закрывает один глаз',expected_appearance:'Сергал'});
    await editor.close(); checks++;
    let loading = await fresh(true, 'pending');
    assert.equal(await loading.locator('#appearance').textContent(), 'Загружаю облик…');
    assert.equal(await loading.locator('#appearance-empty-hint').isVisible(), false);
    await loading.evaluate(() => window.finishProfileLoad());
    await waitClipboard(loading);
    assert.equal(await loading.locator('#appearance').textContent(), 'Сергал');
    assert.equal(await loading.locator('#appearance-empty-hint').isVisible(), false);
    await loading.close(); checks++;
    loading = await fresh(true, 'failed');
    assert.equal(await loading.locator('#appearance').textContent(), 'Не удалось загрузить облик.');
    assert.equal(await loading.locator('#appearance-empty-hint').isVisible(), false);
    assert.match(await message(loading), /Срок доступа к панели истёк/);
    await loading.close(); checks++;
    for (const [mime, ext] of [['image/png','png'], ['image/jpeg','jpg'], ['image/webp','webp']]) {
      const page = await fresh();
      await page.evaluate(type => window.setClipboard('image',type), mime);
      await page.locator('#paste-appearance').click();
      await waitClipboard(page);
      assert.equal(await selected(page), `Картинка из буфера.${ext}`);
      await page.waitForFunction(() => document.querySelector('#appearance-preview').naturalWidth > 0);
      assert.equal(await page.locator('#appearance-preview').isVisible(), true);
      assert.equal(await page.evaluate(() => window.calls.length), 1, 'Pasting must not upload');
      await page.locator('#appearance-species').fill('Сергал');
      await page.locator('#apply-appearance').click();
      await page.waitForFunction(() => document.querySelector('#appearance-status').textContent.includes('сохранён'));
      const calls = await page.evaluate(() => window.calls);
      assert.equal(calls.length, 2);
      assert.equal(calls[1].method, 'POST');
      assert.equal(calls[1].url, '/api/profile/appearance');
      assert.equal(calls[1].type, mime);
      assert.equal(calls[1].size, Buffer.from(png,'base64').length);
      assert.equal(JSON.parse(decodeURIComponent(calls[1].headers['X-Appearance-Options'])).species, 'Сергал');
      assert.equal(await page.locator('#appearance-preview').isVisible(), false);
      await page.waitForFunction(() => document.querySelector('#saved-appearance-image').naturalWidth > 0);
      assert.equal(await page.locator('#saved-appearance').isVisible(), true);
      assert.equal(await page.locator('#appearance-empty-hint').isVisible(), false);
      await page.close(); checks++;
    }
    let page = await fresh();
    assert.equal(await page.evaluate(() => window.paste()), true);
    assert.equal(await selected(page), 'pasted.png');
    assert.equal(await page.evaluate(() => window.calls.length), 1);
    assert.equal(await page.evaluate(() => window.paste('#character')), false);
    assert.equal(await selected(page), 'pasted.png');
    assert.equal(await page.evaluate(() => window.paste('#appearance-input-area','image/gif')), true);
    assert.match(await message(page), /JPEG, PNG или WebP/);
    assert.equal(await selected(page), 'pasted.png', 'Invalid image preserves prior selection');
    await page.evaluate(() => window.paste('#appearance-input-area','image/png',20*1024*1024+1));
    assert.match(await message(page), /до 20 МБ/);
    assert.equal(await selected(page), 'pasted.png');
    await page.evaluate(() => {
      const data=new DataTransfer();data.items.add(window.makeImage('image/png','dropped.png'));
      document.querySelector('#appearance-input-area').dispatchEvent(new DragEvent('drop',{dataTransfer:data,bubbles:true,cancelable:true}));
    });
    assert.equal(await selected(page), 'dropped.png');
    await page.locator('#tab-tools').click();
    assert.equal(await page.evaluate(() => window.paste('#download-url')), false);
    assert.equal(await selected(page), 'dropped.png');
    await page.locator('#tab-rp').click();
    await page.locator('#appearance-mode').selectOption('text');
    assert.equal(await page.locator('#paste-appearance').isVisible(), false);
    assert.equal(await selected(page), '');
    assert.equal(await page.evaluate(() => window.paste()), false);
    await page.locator('#appearance-description').fill('Сергал, голубая шерсть.');
    assert.equal(await page.locator('#apply-appearance').isDisabled(), false);
    await page.close(); checks+=7;

    for (const [mime, ext] of [['image/png','png'], ['image/jpeg','jpg'], ['image/webp','webp']]) {
      page = await fresh(true);
      await page.evaluate(() => window.paste());
      const deltaPreview = await page.locator('#appearance-preview').getAttribute('src');
      await page.locator('#character-upload summary').click();
      await page.evaluate(type => window.setClipboard('image',type), mime);
      await page.locator('#paste-character').click();
      await page.waitForFunction(() => !document.querySelector('#paste-character').disabled);
      assert.equal(await page.locator('#character-file-name').textContent(), `Картинка из буфера.${ext}`);
      await page.waitForFunction(() => document.querySelector('#character-preview').naturalWidth > 0);
      assert.equal(await page.locator('#appearance-preview').getAttribute('src'), deltaPreview);
      assert.equal(await page.evaluate(() => window.calls.length), 1, 'Selecting an image never uploads it');
      await page.locator('#character-species').fill('Дракон');
      await page.locator('#character-notes').fill('Справа');
      await page.locator('#analyze-character').click();
      await page.waitForFunction(() => document.querySelector('#character-status').textContent.startsWith('Твой облик сохранён'));
      const calls = await page.evaluate(() => window.calls);
      assert.equal(calls[1].url, '/api/profile/character/appearance');
      assert.equal(calls[1].type, mime);
      assert.deepEqual(JSON.parse(decodeURIComponent(calls[1].headers['X-Appearance-Options'])), {species:'Дракон',notes:'Справа'});
      assert.equal(await page.locator('#character').inputValue(), 'Новый персонаж');
      assert.equal(await page.locator('#appearance').textContent(), 'Сергал');
      assert.equal(await selected(page), 'pasted.png');
      assert.equal(await page.locator('#character-preview').isVisible(), false);
      assert.equal(await page.locator('#saved-character').isVisible(), true);
      await page.close();checks++;
    }
    page = await fresh();
    await page.locator('#character-upload summary').click();
    assert.equal(await page.evaluate(() => window.paste('#character-input-area')), true);
    assert.equal(await page.locator('#character-file-name').textContent(), 'pasted.png');
    assert.equal(await selected(page), '');
    assert.equal(await page.evaluate(() => window.paste('#character')), false, 'Text fields keep ordinary paste');
    await page.evaluate(() => window.paste('#character-input-area','image/gif'));
    assert.match(await page.locator('#character-status').textContent(), /JPEG, PNG или WebP/);
    assert.equal(await page.locator('#character-file-name').textContent(), 'pasted.png');
    await page.evaluate(() => window.paste('#character-input-area','image/png',20*1024*1024+1));
    assert.match(await page.locator('#character-status').textContent(), /до 20 МБ/);
    await page.evaluate(() => {
      const data=new DataTransfer();data.items.add(window.makeImage('image/png','user-dropped.png'));
      document.querySelector('#character-input-area').dispatchEvent(new DragEvent('drop',{dataTransfer:data,bubbles:true,cancelable:true}));
    });
    assert.equal(await page.locator('#character-file-name').textContent(), 'user-dropped.png');
    await page.locator('#character').fill('Несохранённое описание');
    await page.evaluate(() => {window.failStatus=502;window.failMessage='Ошибка разбора';});
    await page.locator('#analyze-character').click();
    await page.waitForFunction(() => document.querySelector('#character-status').textContent==='Ошибка разбора');
    assert.equal(await page.locator('#character-file-name').textContent(), 'user-dropped.png');
    assert.equal(await page.locator('#character-preview').isVisible(), true);
    assert.equal(await page.locator('#character').inputValue(), 'Несохранённое описание');
    await page.close();checks+=5;
    for (const mode of ['denied','text','unavailable']) {
      page = await fresh();
      await page.locator('#character-upload summary').click();
      await page.evaluate(() => window.paste('#character-input-area'));
      await page.evaluate(mode => {
        if(mode==='unavailable')Object.defineProperty(navigator,'clipboard',{configurable:true,value:undefined});
        else window.setClipboard(mode);
      },mode);
      await page.locator('#paste-character').click();
      await page.waitForFunction(() => !document.querySelector('#paste-character').disabled);
      assert.match(await page.locator('#character-status').textContent(), mode==='text' ? /Скопируй само изображение/ : /Ctrl\+V/);
      assert.equal(await page.locator('#character-file-name').textContent(), 'pasted.png');
      assert.equal(await selected(page), '');
      await page.close();checks++;
    }
    page = await fresh();
    await page.locator('#character-upload summary').click();
    await page.evaluate(() => window.setClipboard('pending'));
    await page.locator('#paste-character').click();
    await page.waitForFunction(() => Boolean(window.finishRead));
    assert.equal(await page.locator('#paste-appearance').isDisabled(), true);
    assert.equal(await page.evaluate(() => window.paste('#appearance-input-area')), false);
    await page.locator('#character-upload summary').click();
    await page.evaluate(() => window.finishRead());
    await waitClipboard(page);
    assert.equal(await page.locator('#character-file-name').textContent(), '');
    assert.equal(await selected(page), '');
    assert.match(await page.locator('#character-status').textContent(), /Вставка отменена/);
    await page.close();checks++;

    page = await fresh();
    await page.locator('#character-upload summary').click();
    assert.equal(await page.evaluate(() => window.paste('body')), true);
    assert.equal(await page.locator('#character-file-name').textContent(), 'pasted.png');
    assert.equal(await selected(page), '');
    await page.locator('#appearance-input-area').focus();
    assert.equal(await page.evaluate(() => window.paste('body')), true);
    assert.equal(await selected(page), 'pasted.png');
    assert.equal(await page.locator('#character-file-name').textContent(), 'pasted.png');
    await page.close();checks++;

    for (const mode of ['denied','text','unavailable']) {
      page = await fresh();
      await page.evaluate(() => window.paste());
      await page.evaluate(mode => {
        if(mode==='unavailable')Object.defineProperty(navigator,'clipboard',{configurable:true,value:undefined});
        else window.setClipboard(mode);
      }, mode);
      await page.locator('#paste-appearance').click();
      await waitClipboard(page);
      assert.match(await message(page), mode==='text' ? /Скопируй само изображение/ : /Ctrl\+V/);
      assert.equal(await selected(page), 'pasted.png');
      assert.equal(await page.evaluate(() => window.calls.length), 1);
      await page.close(); checks++;
    }
    page = await fresh();
    await page.evaluate(() => window.setClipboard('pending'));
    await page.locator('#paste-appearance').click();
    await page.waitForFunction(() => Boolean(window.finishRead));
    assert.equal(await page.locator('#apply-appearance').isDisabled(), true);
    await page.locator('#tab-tools').click();
    await page.evaluate(() => window.finishRead());
    await waitClipboard(page);
    assert.equal(await selected(page), '', 'Discard delayed clipboard after leaving RP tab');
    assert.match(await message(page), /Вставка отменена/);
    await page.close(); checks++;

    page = await fresh(true);
    await page.waitForFunction(() => document.querySelector('#saved-appearance-image').naturalWidth > 0);
    assert.equal(await page.locator('#saved-appearance').isVisible(), true, 'Saved thumbnail restores on opening');
    assert.equal(await page.locator('#appearance-empty-hint').isVisible(), false);
    assert.equal(await page.evaluate(() => document.querySelector('#saved-appearance').compareDocumentPosition(document.querySelector('#appearance-mode')) & Node.DOCUMENT_POSITION_FOLLOWING), 4);
    await page.evaluate(() => {window.paste();window.failStatus=401;window.failMessage='initData просрочен';});
    await page.locator('#apply-appearance').click();
    await page.waitForFunction(() => document.querySelector('#appearance-status').className==='error');
    assert.equal(await page.locator('#saved-appearance-image').getAttribute('src'), await page.evaluate(() => window.thumbnail));
    assert.equal(await selected(page), 'pasted.png');
    await page.evaluate(() => {window.failStatus=0;});
    await page.locator('#reset-appearance').click();
    await page.waitForFunction(() => document.querySelector('#appearance-status').textContent.includes('возвращён'));
    assert.equal(await page.locator('#saved-appearance').isVisible(), false);
    assert.equal(await page.locator('#appearance-empty-hint').isVisible(), true);
    assert.equal(await page.locator('#appearance').textContent(), 'Базовый облик');
    assert.equal(await selected(page), '');
    await page.close(); checks+=3;

    for (const response of [[401,'initData просрочен',/Срок доступа к панели истёк/], [401,'подпись initData не совпала',/открой заново кнопкой бота/], [429,'Подожди 10 секунд',/Подожди 10 секунд/]]) {
      page = await fresh();
      await page.evaluate(() => window.paste());
      await page.evaluate(([status,message]) => {window.failStatus=status;window.failMessage=message;}, response.slice(0,2));
      await page.locator('#apply-appearance').click();
      await page.waitForFunction(() => document.querySelector('#appearance-status').className==='error');
      assert.match(await message(page), response[2]);
      assert.equal(await selected(page), 'pasted.png');
      assert.equal(await page.locator('#appearance-preview').isVisible(), true);
      assert.equal(await page.locator('#appearance').textContent(), 'Базовый облик');
      await page.close(); checks++;
    }
    page = await fresh(true);
    await page.locator('#character-upload summary').click();
    for (const theme of ['dark','light']) {
      await page.evaluate(theme => {
        if(theme==='light')document.body.style.cssText='--tg-theme-text-color:#222;--tg-theme-bg-color:#fff;--tg-theme-hint-color:#667;--tg-theme-secondary-bg-color:#f2f3f5;--tg-theme-button-color:#538cc0';
        else document.body.style.cssText='';
      }, theme);
      for (const width of [360,430,1024]) {
        await page.setViewportSize({width,height:932});
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth>innerWidth), false, 'No horizontal overflow');
        const box=await page.locator('#paste-appearance').boundingBox();
        assert.ok(box.width>100 && box.height>=54 && box.x>=0 && box.x+box.width<=width);
        const characterBox=await page.locator('#paste-character').boundingBox();
        assert.ok(characterBox.width>100 && characterBox.height>=54 && characterBox.x>=0 && characterBox.x+characterBox.width<=width);
        if(process.env.MINIAPP_SCREENSHOTS) {
          fs.mkdirSync(process.env.MINIAPP_SCREENSHOTS,{recursive:true});
          await page.locator('section[aria-labelledby="appearance-title"]').screenshot({path:path.join(process.env.MINIAPP_SCREENSHOTS,`clipboard-${theme}-${width}.png`)});
          await page.locator('#character-upload').screenshot({path:path.join(process.env.MINIAPP_SCREENSHOTS,`character-clipboard-${theme}-${width}.png`)});
        }
        checks++;
      }
    }
    await page.close();
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({browser_checks:checks,page_errors:0,real_clipboard_access:0,external_requests:0}));
  } finally { await browser.close(); }
}
main().catch(error => {console.error(error);process.exitCode=1;});
