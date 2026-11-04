(() => {
  'use strict';
  const app=window.Telegram?.WebApp;
  const controls=['market','weather'].map(name=>({
    button:document.getElementById(`${name}-fullscreen`),label:document.getElementById(`${name}-fullscreen-label`),note:document.getElementById(`${name}-fullscreen-note`),
  })).filter(c=>c.button&&c.label&&c.note);
  const native=Boolean(app?.initData&&app.isVersionAtLeast?.('8.0')&&app.requestFullscreen&&app.exitFullscreen);
  let pending=null;
  function sync(){
    const active=Boolean(document.fullscreenElement||(native&&app.isFullscreen));
    for(const c of controls){c.button.setAttribute('aria-pressed',String(active));c.label.textContent=active?'Свернуть экран':'На весь экран';c.button.title=active?'Выйти из полноэкранного режима':'Развернуть Mini App на весь экран';}
  }
  function complete(){clearTimeout(pending);pending=null;for(const c of controls)c.button.disabled=false;sync();window.dispatchEvent(new Event('resize'));}
  function failed(){complete();for(const c of controls)c.note.textContent='Полный экран недоступен. Попробуй развернуть окно Telegram или обновить приложение.';}
  for(const c of controls)c.button.addEventListener('click',async()=>{
    if(pending!==null)return;
    for(const item of controls){item.note.textContent='';item.button.disabled=true;}
    try{
      if(document.fullscreenElement){await document.exitFullscreen();complete();}
      else if(native){pending=setTimeout(failed,5000);if(app.isFullscreen)app.exitFullscreen();else app.requestFullscreen();}
      else if(document.fullscreenEnabled&&document.documentElement.requestFullscreen){await document.documentElement.requestFullscreen();complete();}
      else{app?.expand?.();failed();}
    }catch{failed();}
  });
  document.addEventListener('fullscreenchange',()=>{for(const c of controls)c.note.textContent='';complete();});
  document.addEventListener('fullscreenerror',failed);
  if(native){app.onEvent('fullscreenChanged',()=>{for(const c of controls)c.note.textContent='';complete();});app.onEvent('fullscreenFailed',failed);app.onEvent('activated',sync);}
  sync();
})();
