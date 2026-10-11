// Synthetic profile/topic responses only; no Telegram, clipboard or model calls.
// Run with Playwright installed: node scripts/check_miniapp_profile_topics.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

const html = fs.readFileSync(path.join(__dirname, '../src/protogen_delta/miniapp/static/index.html'), 'utf8');
const blank = {personality: '', behavior: '', preferences: '', fetishes: '', boundaries: ''};
const profile = {
  user: {first_name: 'Тест', id: 42}, content_mode: 'soft', roleplay_active: false,
  roleplay_configuration: 'male', roleplay_character: '', roleplay_preferences: 'Мои темы',
  roleplay_boundaries: '', roleplay_fetishes: ['chat_topic'], appearance_upload_enabled: true,
  delta_appearance: 'Сохранённый облик', delta_species: null,
  delta_appearance_profile: {...blank, fetishes: 'комета'},
  delta_appearance_topics: [{key: 'topic_a', name: 'Астрономия'}],
};

async function main() {
  const browser = await chromium.launch({headless: true, channel: process.env.MINIAPP_BROWSER_CHANNEL || undefined});
  const context = await browser.newContext({viewport: {width: 430, height: 932}});
  await context.route('**/*', route => route.abort());
  const errors = [];
  let checks = 0;
  try {
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    const setup = `<script>
      window.Telegram={WebApp:{initData:'synthetic',ready(){},expand(){}}};
      window.testProfile=${JSON.stringify(profile)};
      window.failSave=false;window.calls=[];
      window.fetch=async(url,options)=>{
        const body=options.body ? JSON.parse(options.body) : null;
        window.calls.push({url,method:options.method,body});
        if(url==='/api/profile/appearance/behavior' && options.method==='PATCH') {
          if(window.failSave)return {ok:false,status:409,text:async()=>'Профиль уже изменился'};
          window.testProfile.delta_appearance_profile=body.profile;
          window.testProfile.delta_appearance_topics=body.profile.fetishes==='комета и загадка'
            ? [{key:'topic_a',name:'Астрономия'},{key:'topic_b',name:'Загадки <b>текст</b>'}] : [];
        }
        return {ok:true,json:async()=>structuredClone(window.testProfile)};
      };
    </script>`;
    await page.setContent(html.replace(/<script src="https:\/\/telegram.org[^>]+><\/script>/, setup));
    await page.waitForFunction(() => !document.querySelector('#delta-fetishes').disabled);
    const hint = page.locator('#delta-fetish-topics');
    assert.equal(await hint.textContent(), 'Распознанные темы: Астрономия.');
    assert.equal(await page.locator('#delta-fetishes').inputValue(), 'комета');
    checks++;

    await page.locator('#delta-fetishes').fill('комета и загадка');
    assert.equal(await hint.textContent(), 'Распознанные темы обновятся после сохранения.');
    await page.evaluate(() => {window.failSave = true;});
    await page.locator('#save-delta-profile').click();
    await page.waitForFunction(() => document.querySelector('#delta-profile-status').textContent === 'Профиль уже изменился');
    assert.equal(await page.locator('#delta-fetishes').inputValue(), 'комета и загадка');
    assert.equal(await hint.textContent(), 'Распознанные темы обновятся после сохранения.');
    checks++;

    await page.evaluate(() => {window.failSave = false;});
    await page.locator('#save-delta-profile').click();
    await page.waitForFunction(() => document.querySelector('#delta-profile-status').textContent.includes('сохранены'));
    assert.equal(await hint.textContent(), 'Распознанные темы: Астрономия, Загадки <b>текст</b>.');
    assert.equal(await hint.locator('b').count(), 0);
    assert(await page.locator('#save-delta-profile').isDisabled());
    const savedRequest = await page.evaluate(() => window.calls.filter(call => call.url === '/api/profile/appearance/behavior').at(-1));
    assert.equal(savedRequest.body.profile.fetishes, 'комета и загадка');
    assert.equal(savedRequest.body.expected_profile.fetishes, 'комета');
    assert.equal(Object.hasOwn(savedRequest.body, 'delta_appearance_topics'), false);
    assert.equal(await page.locator('#preferences').inputValue(), 'Мои темы');
    assert.deepEqual(await page.evaluate(() => window.testProfile.roleplay_fetishes), ['chat_topic']);
    checks++;

    await page.locator('#delta-fetishes').fill('Неизвестный текст');
    await page.locator('#save-delta-profile').click();
    await page.waitForFunction(() => document.querySelector('#save-delta-profile').disabled);
    assert.equal(await hint.textContent(), 'Распознанные темы: пока нет.');
    assert.equal(await page.locator('#delta-fetishes').inputValue(), 'Неизвестный текст');
    checks++;

    await page.locator('#delta-fetishes').fill('');
    await page.locator('#save-delta-profile').click();
    await page.waitForFunction(() => document.querySelector('#save-delta-profile').disabled);
    assert.equal(await hint.textContent(), 'Распознанные темы: пока нет.');
    assert.equal(await page.locator('#delta-fetishes').inputValue(), '');
    checks++;

    for (const width of [360, 430, 1024]) {
      await page.setViewportSize({width, height: 932});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      checks++;
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({checks, browser_errors: 0, external_requests: 0}));
  } finally {
    await browser.close();
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
