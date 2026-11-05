(() => {
  'use strict';
  const $=id=>document.getElementById(id), time=window.DeltaWeatherTime;
  const layers={precip:'Осадки',tmp2m:'Температура',wind:'Ветер',pressure:'Давление'};
  let map=null,styleReady=false,marker=null,point=null,layer='precip',base=0,step=0,revision=0,sourceId=null,frameReady=false,failed=false,playing=null;
  const cache=new Map(), fields=new Map(), pending=new Map();
  let rain=null,rainContext=null,rainAnimation=0,rainTick=0,drops=[],wetCells=[];
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  const tileKey=(stamp,z,x,y)=>`${stamp}/${z}/${x}/${y}`;
  let requests=0;const waiting=[];
  async function limited(task){
    if(requests>=6)await new Promise(resolve=>waiting.push(resolve));requests++;
    try{return await task();}finally{requests--;waiting.shift()?.();}
  }
  function remember(store,key,value,limit=96){store.delete(key);store.set(key,value);while(store.size>limit)store.delete(store.keys().next().value);}
  async function field(stamp,z,x,y,signal){
    signal.throwIfAborted();
    const count=2**z;x=(x%count+count)%count;y=Math.max(0,Math.min(count-1,y));
    const key=tileKey(stamp,z,x,y);
    if(fields.has(key))return fields.get(key);
    let entry=pending.get(key);
    if(!entry||entry.controller.signal.aborted){
      entry={controller:new AbortController(),consumers:0,settled:false};const current=entry;
      entry.task=limited(async()=>{
      current.controller.signal.throwIfAborted();
      const response=await fetch(`https://weathermaps.weatherapi.com/precip/tiles/${stamp}/${z}/${x}/${y}.png`,{signal:AbortSignal.any([current.controller.signal,AbortSignal.timeout(10000)])});
      if(!response.ok)throw new Error('Данные карты на этот час недоступны');
      const bytes=await response.arrayBuffer();if(bytes.byteLength>1048576)throw new Error('Слишком большой участок карты');
      const image=await createImageBitmap(new Blob([bytes],{type:'image/png'}));
      if(image.width!==256||image.height!==256){image.close();throw new Error('Неверный размер участка карты');}
      const canvas=document.createElement('canvas');canvas.width=256;canvas.height=256;
      const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(image,0,0);image.close();
      const pixels=ctx.getImageData(0,0,256,256).data,values=new Uint8Array(256*256);
      for(let i=0;i<values.length;i++)values[i]=Math.round((255-Math.min(pixels[i*4],pixels[i*4+1],pixels[i*4+2]))*pixels[i*4+3]/255);
      current.controller.signal.throwIfAborted();remember(fields,key,values);return values;
      }).finally(()=>{current.settled=true;if(pending.get(key)===current)pending.delete(key);});
      pending.set(key,entry);
    }
    entry.consumers++;let released=false;
    const release=()=>{if(released)return;released=true;entry.consumers--;if(!entry.consumers&&!entry.settled)entry.controller.abort();};
    signal.addEventListener('abort',release,{once:true});
    try{return await entry.task;}finally{signal.removeEventListener('abort',release);release();}
  }
  async function precipitation(stamp,z,x,y,signal){
    // Соседние данные дают одинаковое сглаживание по обе стороны границы тайла.
    const neighbors=await Promise.all(Array.from({length:9},(_,i)=>field(stamp,z,x+i%3-1,y+Math.floor(i/3)-1,signal)));
    signal.throwIfAborted();
    const pad=8,size=256+pad*2,canvas=document.createElement('canvas');canvas.width=size;canvas.height=size;
    const ctx=canvas.getContext('2d'),input=new Float32Array(size*size),horizontal=new Float32Array(size*size);
    for(let py=0;py<size;py++)for(let px=0;px<size;px++){
      const sx=px-pad,sy=py-pad,nx=Math.floor(sx/256),ny=Math.floor(sy/256);
      input[py*size+px]=neighbors[(ny+1)*3+nx+1][((sy+256)%256)*256+(sx+256)%256]/255;
    }
    // Свёртка поля до раскраски: без ступеней палитры и зависимости от Canvas.filter.
    const radius=6,kernel=Array.from({length:radius*2+1},(_,i)=>Math.exp(-((i-radius)**2)/(2*2.5**2))),sum=kernel.reduce((a,b)=>a+b);
    for(let i=0;i<kernel.length;i++)kernel[i]/=sum;
    for(let py=0;py<size;py++)for(let px=pad-1;px<=size-pad;px++){
      let value=0;for(let k=-radius;k<=radius;k++)value+=input[py*size+px+k]*kernel[k+radius];horizontal[py*size+px]=value;
    }
    const coloredSize=258,values=ctx.createImageData(coloredSize,coloredSize),ramp=[[0,67,106,157,0],[0.12,67,106,157,110],[0.4,50,88,150,172],[1,29,54,115,220]];
    for(let i=0;i<values.data.length;i+=4){
      const px=(i/4)%coloredSize+pad-1,py=Math.floor(i/4/coloredSize)+pad-1;let strength=0;
      for(let k=-radius;k<=radius;k++)strength+=horizontal[(py+k)*size+px]*kernel[k+radius];
      let segment=1;while(segment<ramp.length-1&&strength>ramp[segment][0])segment++;
      const a=ramp[segment-1],b=ramp[segment],t=(strength-a[0])/(b[0]-a[0]);
      for(let c=0;c<4;c++)values.data[i+c]=Math.round(a[c+1]+(b[c+1]-a[c+1])*t);
    }
    canvas.width=coloredSize;canvas.height=coloredSize;ctx.putImageData(values,0,0);
    const enlarged=document.createElement('canvas');enlarged.width=516;enlarged.height=516;
    const enlargedContext=enlarged.getContext('2d');enlargedContext.imageSmoothingEnabled=true;enlargedContext.imageSmoothingQuality='high';enlargedContext.drawImage(canvas,0,0,516,516);
    const output=document.createElement('canvas');output.width=512;output.height=512;
    output.getContext('2d').putImageData(enlargedContext.getImageData(2,2,512,512),0,0);
    const blob=await new Promise(resolve=>output.toBlob(resolve,'image/png'));
    if(!blob)throw new Error('Не удалось подготовить карту');return blob.arrayBuffer();
  }
  function wetAt(px,py){
    const location=map.unproject([px,py]),sin=Math.sin(location.lat*Math.PI/180);
    const mx=(location.lng/360+0.5)%1,my=(1-Math.log((1+sin)/(1-sin))/(2*Math.PI))/2;
    const stamp=time.utcStamp(base+step*3600000);
    for(let z=Math.min(6,Math.floor(map.getZoom()));z>=0;z--){
      const n=2**z,wx=((mx+1)%1)*n,wy=Math.max(0,Math.min(n-1e-8,my*n)),x=Math.floor(wx),y=Math.floor(wy);
      const values=fields.get(tileKey(stamp,z,x,y));
      if(values)return values[Math.floor((wy-y)*256)*256+Math.floor((wx-x)*256)]>0;
    }
    return false;
  }
  function clearRain(){cancelAnimationFrame(rainAnimation);rainAnimation=0;rainTick=0;drops=[];rainContext?.clearRect(0,0,rain.width,rain.height);}
  function rebuildRain(){
    clearRain();wetCells=[];
    if(!rain||!frameReady||failed||layer!=='precip'||motion.matches||!$('weather-rain').checked||document.hidden||$('weather-panel').hidden)return;
    const width=rain.clientWidth,height=rain.clientHeight,scale=Math.min(devicePixelRatio||1,1.5);
    if(!width||!height)return;rain.width=Math.round(width*scale);rain.height=Math.round(height*scale);rainContext.setTransform(scale,0,0,scale,0,0);
    for(let y=12;y<height;y+=24)for(let x=12;x<width;x+=24)if(wetAt(x,y))wetCells.push([x,y]);
    if(!wetCells.length)return;
    // Частицы — условное обозначение осадков; их число не изображает мм/ч или ветер.
    const count=Math.min(110,Math.ceil(wetCells.length/4));
    const spawn=()=>{const p=wetCells[Math.floor(Math.random()*wetCells.length)];return {x:p[0],y:p[1],age:0,life:0.4+Math.random()*0.6,speed:13+Math.random()*12};};
    drops=Array.from({length:count},()=>{const p=spawn();p.age=Math.random()*p.life;return p;});
    const draw=tick=>{
      const delta=Math.min(0.07,(tick-rainTick)/1000||0);if(rainTick&&tick-rainTick<32){rainAnimation=requestAnimationFrame(draw);return;}rainTick=tick;
      rainContext.clearRect(0,0,width,height);rainContext.lineWidth=1;rainContext.lineCap='round';
      for(let i=0;i<drops.length;i++){
        const p=drops[i];p.age+=delta;p.y+=p.speed*delta;
        if(p.age>p.life){drops[i]=spawn();continue;}
        if(!wetAt(p.x,p.y))continue;
        rainContext.strokeStyle=`rgba(198,220,247,${0.32*Math.sin(p.age/p.life*Math.PI)})`;
        rainContext.beginPath();rainContext.moveTo(p.x,p.y);rainContext.lineTo(p.x-0.6,p.y+4);rainContext.stroke();
      }
      rainAnimation=requestAnimationFrame(draw);
    };
    rainAnimation=requestAnimationFrame(draw);
  }
  const labels={
    'NavigationControl.ZoomIn':'Приблизить','NavigationControl.ZoomOut':'Отдалить','NavigationControl.ResetBearing':'На север',
    'AttributionControl.ToggleAttribution':'Источники карты','Map.ScrollZoomBlocker.CtrlMessage':'Удерживай Ctrl для изменения масштаба',
    'Map.ScrollZoomBlocker.CmdMessage':'Удерживай ⌘ для изменения масштаба',
    'Map.CooperativeGesturesHandler.TouchPanBlockerMessage':'Перемещай карту двумя пальцами',
  };
  const note=text=>{$('weather-map-status').textContent=text;};
  const fullTime=()=>time.full(base+step*3600000);
  let lightningOn=false,lightningController=null,lightningTimer=null,lightningRevision=0,lightningPopup=null;
  function clearLightning(){
    clearTimeout(lightningTimer);lightningTimer=null;lightningController?.abort();lightningController=null;lightningRevision++;
    lightningPopup?.remove();lightningPopup=null;
    map?.getSource('weather-lightning')?.setData({type:'FeatureCollection',features:[]});
  }
  async function lightning(){
    clearLightning();
    if(!lightningOn||!point||!styleReady)return;
    const status=$('weather-lightning-status');
    if(step!==0){status.textContent='Разряды молний — наблюдения за последние 30 минут. Верни шкалу на «Сейчас».';return;}
    if(document.hidden||$('weather-panel').hidden)return;
    const initData=window.Telegram?.WebApp?.initData;
    if(!initData){status.textContent='Открой миниапп внутри Telegram.';return;}
    const current=lightningRevision,controller=new AbortController();lightningController=controller;
    const timeout=setTimeout(()=>controller.abort(),12000);status.textContent='Загружаю наблюдения за молниями…';
    try{
      const query=new URLSearchParams({latitude:point.latitude,longitude:point.longitude});
      const response=await fetch(`/api/weather/lightning?${query}`,{headers:{'X-Telegram-Init-Data':initData},signal:controller.signal});
      if(!response.ok)throw new Error('Наблюдения за молниями сейчас недоступны. Попробуй позже.');
      const data=await response.json();if(current!==lightningRevision)return;
      if(data.status==='disabled'){lightningOn=false;$('weather-lightning').setAttribute('aria-pressed','false');status.textContent='Наблюдения за молниями пока не подключены.';return;}
      if(data.status==='outside_coverage'){status.textContent='Для этого города нет покрытия наблюдений за молниями. Это не означает отсутствия грозы.';return;}
      if(data.status!=='ok'||!Array.isArray(data.strikes))throw new Error('Источник молний вернул неполные данные.');
      const features=data.strikes.map(s=>({type:'Feature',geometry:{type:'Point',coordinates:[s.longitude,s.latitude]},properties:{time:s.time}}));
      if(!map.getSource('weather-lightning')){
        map.addSource('weather-lightning',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
        map.addLayer({id:'weather-lightning-points',type:'circle',source:'weather-lightning',paint:{'circle-color':'#ffd56b','circle-radius':4,'circle-stroke-color':'#c77931','circle-stroke-width':1.5}});
      }
      map.getSource('weather-lightning').setData({type:'FeatureCollection',features});
      status.textContent=`Foreca · разряды за 30 минут в радиусе 100 км: ${features.length}${data.possibly_truncated?' (выборка ограничена)':''}. Данные на ${time.full(data.as_of)}. Погрешность положения около 8–15 км, задержка 5–15 минут.`;
      lightningTimer=setTimeout(lightning,600000);
    }catch(error){if(current===lightningRevision)status.textContent=error.name==='AbortError'?'Источник молний не ответил вовремя. Попробуй позже.':error.message;}
    finally{clearTimeout(timeout);if(lightningController===controller)lightningController=null;}
  }
  function stop(){clearInterval(playing);playing=null;$('weather-play').textContent='▶';$('weather-play').setAttribute('aria-pressed','false');$('weather-play').title='Показать движение прогноза';}
  function updateTime(){
    $('weather-time').value=String(step);$('weather-map-time').textContent=fullTime();
    $('weather-time').setAttribute('aria-valuetext',fullTime());
  }
  async function tiles(request,abort){
    const match=/^weather:\/\/(precip|tmp2m|wind|pressure)\/(\d{10})\/(\d{1,2})\/(\d+)\/(\d+)\.png$/.exec(request.url);
    if(!match)throw new Error('Некорректный адрес погодных данных');
    const [,kind,stamp,z,x,y]=match;
    if(+z>6||+x>=2**+z||+y>=2**+z)throw new Error('Некорректный участок карты');
    if(cache.has(request.url)){const data=cache.get(request.url);cache.delete(request.url);cache.set(request.url,data);return {data:data.slice(0)};}
    if(kind==='precip'){
      const data=await precipitation(stamp,+z,+x,+y,abort.signal);remember(cache,request.url,data);return {data:data.slice(0)};
    }
    const response=await fetch(`https://weathermaps.weatherapi.com/${kind}/tiles/${stamp}/${z}/${x}/${y}.png`,{signal:abort.signal});
    if(!response.ok)throw new Error('Данные карты на этот час недоступны');
    let data=await response.arrayBuffer();
    if(data.byteLength>1048576)throw new Error('Слишком большой участок карты');
    cache.set(request.url,data);while(cache.size>96)cache.delete(cache.keys().next().value);
    return {data:data.slice(0)};
  }
  if(window.maplibregl)maplibregl.addProtocol('weather',tiles);
  function frame(){
    updateTime();if(!map||!styleReady)return;
    revision++;frameReady=false;failed=false;clearRain();lightning();
    if(sourceId){if(map.getLayer(sourceId))map.removeLayer(sourceId);if(map.getSource(sourceId))map.removeSource(sourceId);}
    sourceId=`weather-frame-${revision}`;
    const stamp=time.utcStamp(base+step*3600000);
    map.addSource(sourceId,{type:'raster',tiles:[`weather://${layer}/${stamp}/{z}/{x}/{y}.png`],tileSize:256,minzoom:0,maxzoom:6,attribution:'<a href="https://www.weatherapi.com/" target="_blank" rel="noopener">WeatherAPI</a>'});
    const firstLabel=map.getStyle().layers.find(l=>l.type==='symbol')?.id;
    map.addLayer({id:sourceId,type:'raster',source:sourceId,paint:{'raster-opacity':layer==='precip'?0.95:0.6,'raster-fade-duration':180,'raster-resampling':'linear'}},firstLabel);
    note(`Загружаю: ${layers[layer].toLowerCase()} · ${fullTime()}`);
    $('weather-map-legend').textContent=layer==='precip'?'Синие зоны и капли обозначают осадки. Границы сглажены; точное количество — в карточках города.':`${layers[layer]}: цветовой прогноз. Точные значения — в карточках города.`;
    $('weather-map-legend').classList.toggle('precip-legend',layer==='precip');
  }
  function create(){
    if(!window.maplibregl){note('Не удалось загрузить карту. Обнови миниапп.');return;}
    try{
      map=new maplibregl.Map({container:'weather-map',style:'/static/weather-map-style.json',center:[point.longitude,point.latitude],zoom:7,minZoom:2,maxZoom:11,dragRotate:false,touchPitch:false,locale:labels,maxTileCacheSize:64});
      rain=document.createElement('canvas');rain.className='weather-rain-canvas';rain.setAttribute('aria-hidden','true');rainContext=rain.getContext('2d');$('weather-map').append(rain);
      map.on('movestart',clearRain);map.on('moveend',rebuildRain);map.on('resize',rebuildRain);
      map.on('click','weather-lightning-points',e=>{
        const feature=e.features?.[0];if(feature&&lightningOn&&step===0){lightningPopup?.remove();lightningPopup=new maplibregl.Popup().setLngLat(e.lngLat).setText(`Разряд молнии · ${time.full(feature.properties.time)} · Foreca`).addTo(map);}
      });
      map.addControl(new maplibregl.NavigationControl({showCompass:false}),'top-right');
      map.on('load',()=>{styleReady=true;marker=new maplibregl.Marker({color:'#f04b4b'}).setLngLat([point.longitude,point.latitude]).addTo(map);frame();});
      map.on('sourcedata',e=>{if(e.sourceId===sourceId&&e.isSourceLoaded&&!failed){frameReady=true;note(`Почасовой модельный прогноз · ${fullTime()}`);rebuildRain();}});
      map.on('error',e=>{
        if(e.error?.name==='AbortError')return;
        if(e.sourceId===sourceId){failed=true;frameReady=false;stop();clearRain();if(map.getLayer(sourceId))map.setLayoutProperty(sourceId,'visibility','none');note('Данные осадков на этот час недоступны. Выбери другое время или повтори позже.');}
        else if(!map.isStyleLoaded()){note('Основа карты пока не загрузилась. Проверь соединение и обнови миниапп.');}
      });
      new ResizeObserver(()=>map.resize()).observe($('weather-map'));
      window.addEventListener('resize',()=>map.resize());
    }catch{note('Карта недоступна в этом окне. Обнови Telegram или открой миниапп на другом устройстве.');}
  }
  window.DeltaWeatherMap={setLocation(data){
    const coords=data.coordinates;
    if(!coords||!Number.isFinite(coords.latitude)||!Number.isFinite(coords.longitude)||Math.abs(coords.latitude)>90||Math.abs(coords.longitude)>180)return;
    stop();point=coords;
    const observed=time.date(data.current?.time);base=Math.floor((observed?.valueOf()||Date.now())/3600000)*3600000;step=0;
    if(!map)create();else{map.jumpTo({center:[point.longitude,point.latitude],zoom:7});marker?.setLngLat([point.longitude,point.latitude]);frame();}
    updateTime();
  }};
  $('weather-time').addEventListener('input',()=>{stop();step=Math.max(0,Math.min(48,Number($('weather-time').value)||0));frame();});
  $('weather-now').addEventListener('click',()=>{stop();step=0;frame();});
  $('weather-play').addEventListener('click',()=>{
    if(playing){stop();return;}if(!map)return;
    $('weather-play').textContent='Ⅱ';$('weather-play').setAttribute('aria-pressed','true');$('weather-play').title='Приостановить';
    playing=setInterval(()=>{if(document.hidden||$('weather-panel').hidden){stop();return;}if(frameReady){step=(step+1)%49;frame();}},1800);
  });
  for(const button of document.querySelectorAll('[data-weather-layer]'))button.addEventListener('click',()=>{
    if(!Object.hasOwn(layers,button.dataset.weatherLayer))return;
    stop();layer=button.dataset.weatherLayer;
    for(const b of document.querySelectorAll('[data-weather-layer]'))b.setAttribute('aria-pressed',String(b===button));frame();
  });
  $('weather-rain').addEventListener('change',rebuildRain);
  $('weather-lightning').addEventListener('click',()=>{lightningOn=!lightningOn;$('weather-lightning').setAttribute('aria-pressed',String(lightningOn));if(!lightningOn){clearLightning();$('weather-lightning-status').textContent='Наблюдения за молниями выключены.';}else lightning();});
  motion.addEventListener('change',rebuildRain);
  new MutationObserver(()=>{rebuildRain();if($('weather-panel').hidden)clearLightning();else lightning();}).observe($('weather-panel'),{attributes:true,attributeFilter:['hidden']});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();clearRain();clearLightning();}else{rebuildRain();lightning();}});
})();
