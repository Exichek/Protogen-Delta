(() => {
  'use strict';
  const $=id=>document.getElementById(id), time=window.DeltaWeatherTime;
  const layers={precip:'Осадки',tmp2m:'Температура',wind:'Ветер',pressure:'Давление'};
  let map=null,styleReady=false,marker=null,point=null,layer='precip',base=0,step=0,revision=0,sourceId=null,frameReady=false,failed=false,playing=null;
  const cache=new Map();
  const labels={
    'NavigationControl.ZoomIn':'Приблизить','NavigationControl.ZoomOut':'Отдалить','NavigationControl.ResetBearing':'На север',
    'AttributionControl.ToggleAttribution':'Источники карты','Map.ScrollZoomBlocker.CtrlMessage':'Удерживай Ctrl для изменения масштаба',
    'Map.ScrollZoomBlocker.CmdMessage':'Удерживай ⌘ для изменения масштаба',
    'Map.CooperativeGesturesHandler.TouchPanBlockerMessage':'Перемещай карту двумя пальцами',
  };
  const note=text=>{$('weather-map-status').textContent=text;};
  const fullTime=()=>time.full(base+step*3600000);
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
    const response=await fetch(`https://weathermaps.weatherapi.com/${kind}/tiles/${stamp}/${z}/${x}/${y}.png`,{signal:abort.signal});
    if(!response.ok)throw new Error('Данные карты на этот час недоступны');
    let data=await response.arrayBuffer();
    if(data.byteLength>1048576)throw new Error('Слишком большой участок карты');
    if(kind==='precip'){
      const image=await createImageBitmap(new Blob([data],{type:'image/png'}));
      if(image.width!==256||image.height!==256){image.close();throw new Error('Неверный размер участка карты');}
      const canvas=document.createElement('canvas');canvas.width=256;canvas.height=256;
      const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(image,0,0);image.close();
      const pixels=ctx.getImageData(0,0,256,256), values=pixels.data;
      for(let i=0;i<values.length;i+=4){
        const strength=(255-Math.min(values[i],values[i+1],values[i+2]))/255;
        if(strength<=0){values[i+3]=0;continue;}
        const color=strength<0.22?[65,105,164,Math.round(70+Math.min(strength,0.22)*320)]:strength<0.55?[43,78,139,185]:[28,52,109,215];
        values[i]=color[0];values[i+1]=color[1];values[i+2]=color[2];values[i+3]=Math.round(color[3]*values[i+3]/255);
      }
      ctx.putImageData(pixels,0,0);
      const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));
      if(!blob)throw new Error('Не удалось подготовить карту');
      data=await blob.arrayBuffer();
    }
    cache.set(request.url,data);while(cache.size>96)cache.delete(cache.keys().next().value);
    return {data:data.slice(0)};
  }
  if(window.maplibregl)maplibregl.addProtocol('weather',tiles);
  function frame(){
    updateTime();if(!map||!styleReady)return;
    revision++;frameReady=false;failed=false;
    if(sourceId){if(map.getLayer(sourceId))map.removeLayer(sourceId);if(map.getSource(sourceId))map.removeSource(sourceId);}
    sourceId=`weather-frame-${revision}`;
    const stamp=time.utcStamp(base+step*3600000);
    map.addSource(sourceId,{type:'raster',tiles:[`weather://${layer}/${stamp}/{z}/{x}/{y}.png`],tileSize:256,minzoom:0,maxzoom:6,attribution:'<a href="https://www.weatherapi.com/" target="_blank" rel="noopener">WeatherAPI</a>'});
    const firstLabel=map.getStyle().layers.find(l=>l.type==='symbol')?.id;
    map.addLayer({id:sourceId,type:'raster',source:sourceId,paint:{'raster-opacity':layer==='precip'?0.95:0.6,'raster-fade-duration':180}},firstLabel);
    note(`Загружаю: ${layers[layer].toLowerCase()} · ${fullTime()}`);
    $('weather-map-legend').textContent=layer==='precip'?'Зоны осадков показаны синим. Заливка условная; точное количество — в карточках города.':`${layers[layer]}: цветовой прогноз. Точные значения — в карточках города.`;
    $('weather-map-legend').classList.toggle('precip-legend',layer==='precip');
  }
  function create(){
    if(!window.maplibregl){note('Не удалось загрузить карту. Обнови миниапп.');return;}
    try{
      map=new maplibregl.Map({container:'weather-map',style:'/static/weather-map-style.json',center:[point.longitude,point.latitude],zoom:7,minZoom:2,maxZoom:11,dragRotate:false,touchPitch:false,locale:labels,maxTileCacheSize:64});
      map.addControl(new maplibregl.NavigationControl({showCompass:false}),'top-right');
      map.on('load',()=>{styleReady=true;marker=new maplibregl.Marker({color:'#f04b4b'}).setLngLat([point.longitude,point.latitude]).addTo(map);frame();});
      map.on('sourcedata',e=>{if(e.sourceId===sourceId&&e.isSourceLoaded&&!failed){frameReady=true;note(`Почасовой модельный прогноз · ${fullTime()}`);}});
      map.on('error',e=>{
        if(e.error?.name==='AbortError')return;
        if(e.sourceId===sourceId){failed=true;frameReady=false;stop();if(map.getLayer(sourceId))map.setLayoutProperty(sourceId,'visibility','none');note('Данные осадков на этот час недоступны. Выбери другое время или повтори позже.');}
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
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
})();
