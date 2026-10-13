// Синтетические профили: настройка речи, сохранение, ошибки и сброс без Telegram/LLM.
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
const html=require('./miniapp_fixture.cjs').readMiniAppHtml();
const profile={user:{id:42,first_name:'Тест'},content_mode:'soft',roleplay_active:false,roleplay_configuration:'male',roleplay_character:'',roleplay_preferences:'',roleplay_boundaries:'',roleplay_fetishes:[],delta_appearance:'',delta_species:null,delta_appearance_topics:[],appearance_upload_enabled:true,delta_appearance_profile:{personality:'',behavior:'',preferences:'',fetishes:'',boundaries:''}};
async function main(){
  const browser=await chromium.launch({headless:true,channel:process.env.MINIAPP_BROWSER_CHANNEL || undefined});
  const context=await browser.newContext({viewport:{width:430,height:800}});const errors=[];let checks=0;
  await context.route('**/*',route=>route.abort());
  try{
    const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));
    const setup=`<script>
      window.Telegram={WebApp:{initData:'synthetic',ready(){},expand(){}}};window.confirm=()=>true;
      window.profile=${JSON.stringify(profile)};window.calls=[];window.failure=false;window.deferred=false;
      window.fetch=async(url,options)=>{
        const body=options.body?JSON.parse(options.body):null;window.calls.push({url,method:options.method,body});
        if(window.failure)return {ok:false,status:409,text:async()=>'Сохранение не удалось'};
        if(options.method==='PATCH'){
          if(window.deferred)await new Promise(resolve=>window.resolveSave=resolve);
          Object.assign(window.profile,body);
        }
        if(options.method==='DELETE')window.profile.roleplay_allow_profanity=false;
        return {ok:true,json:async()=>structuredClone(window.profile)};
      };
    </script>`;
    await page.setContent(html.replace(/<script src="https:\/\/telegram.org[^>]+><\/script>/,setup));
    await page.waitForFunction(()=>document.querySelector('#hello').textContent.includes('42'));
    const checkbox=page.locator('#rp-allow-profanity');assert(!(await checkbox.isChecked()));checks++;
    await checkbox.check();await page.evaluate(()=>{window.failure=true;});await page.locator('#save').click();
    await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Сохранение не удалось'));
    assert(await checkbox.isChecked());assert.equal(await page.evaluate(()=>window.calls.at(-1).body.roleplay_allow_profanity),true);checks++;
    await page.evaluate(()=>{window.failure=false;window.deferred=true;});await page.locator('#save').click();
    await page.waitForFunction(()=>Boolean(window.resolveSave));assert(await checkbox.isDisabled());checks++;
    await page.evaluate(()=>{window.deferred=false;window.resolveSave();});await page.waitForFunction(()=>document.querySelector('#status').textContent==='Сохранено.');
    assert(await checkbox.isChecked());assert(await checkbox.isEnabled());checks++;
    await page.evaluate(async()=>{document.querySelector('#rp-allow-profanity').checked=false;render(await api());});assert(await checkbox.isChecked());checks++;
    await checkbox.uncheck();await page.locator('#save').click();await page.waitForFunction(()=>document.querySelector('#status').textContent==='Сохранено.');
    assert(!(await checkbox.isChecked()));assert.equal(await page.evaluate(()=>window.calls.at(-1).body.roleplay_allow_profanity),false);checks++;
    await checkbox.check();await page.locator('#save').click();await page.waitForFunction(()=>document.querySelector('#status').textContent==='Сохранено.');
    await page.locator('#clear-rp').click();await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('очищен'));
    assert(!(await checkbox.isChecked()));checks++;
    for(const width of [320,430,720,1280]){await page.setViewportSize({width,height:800});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));checks++;}
    assert.equal(errors.length,0,errors.join('\n'));console.log(JSON.stringify({checks,page_errors:0,external_requests:0,real_model_calls:0}));
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
