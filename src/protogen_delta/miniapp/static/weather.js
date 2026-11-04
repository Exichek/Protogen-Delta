(() => {
  'use strict';
  const $=id=>document.getElementById(id), app=window.Telegram?.WebApp;
  const city=$('weather-city'), country=$('weather-country'), status=$('weather-status');
  const number=value=>typeof value==='number' && Number.isFinite(value);
  const display=(value,unit='',digits=0)=>number(value)?`${value.toFixed(digits)}${unit.replace('m/s','м/с').replace('mm','мм').replace('cm','см')}`:'—';
  const icon=code=>!number(code)?'？':code===0?'☀️':code<=2?'🌤️':code===3?'☁️':code<50?'🌫️':code>=95?'⛈️':(code>=71&&code<=77)||code===85||code===86?'🌨️':'🌧️';
  const node=(tag,text,className)=>{const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;};
  let coordinates=null, busy=false;
  function map(){
    if(!coordinates)return;
    const {latitude:lat,longitude:lon}=coordinates;
    if(!number(lat)||!number(lon)||Math.abs(lat)>90||Math.abs(lon)>180)return;
    const overlay=$('weather-layer').value;
    if(!['rain','temp','wind'].includes(overlay))return;
    const params=new URLSearchParams({type:'map',location:'coordinates',lat:String(lat),lon:String(lon),zoom:'7',overlay,product:'ecmwf',level:'surface',metricTemp:'°C',metricWind:'m/s',metricRain:'mm',marker:'true'});
    const url=`https://embed.windy.com/embed.html?${params}`;
    $('weather-map').src=url;$('weather-map-link').href=url;
  }
  function detail(label,value){const el=node('div');el.append(node('span',label,'muted'),node('b',value));$('weather-details').append(el);}
  function render(data){
    const c=data.current||{}, u=data.current_units||{}, h=data.hourly||{}, hu=data.hourly_units||{}, d=data.daily||{}, du=data.daily_units||{};
    $('weather-empty').hidden=true;$('weather-results').hidden=false;
    $('weather-place').textContent=[data.place?.name,data.place?.admin1,data.place?.country].filter(Boolean).join(', ');
    $('weather-temperature').textContent=`${icon(c.weather_code)} ${display(c.temperature_2m,'°')}`;
    $('weather-condition').textContent=data.conditions?.current||'Описание условий недоступно';
    $('weather-feels').textContent=`Ощущается как ${display(c.apparent_temperature,'°')}`;
    $('weather-updated').textContent=`Время данных: ${(c.time||'—').replace('T',' ')} · ${data.timezone||''}`;
    $('weather-details').replaceChildren();
    detail('Ветер',display(c.wind_speed_10m,` ${u.wind_speed_10m||'м/с'}`,1));
    detail('Порывы',display(c.wind_gusts_10m,` ${u.wind_gusts_10m||'м/с'}`,1));
    detail('Влажность',display(c.relative_humidity_2m,'%'));
    detail('Осадки сейчас',display(c.precipitation,` ${u.precipitation||'мм'}`,1));
    detail('Снег сейчас',display(c.snowfall,` ${u.snowfall||'см'}`,1));
    detail('Давление',number(c.surface_pressure)?display(c.surface_pressure*0.750061683,' мм рт. ст.'):'—');
    $('weather-hours').replaceChildren();
    (h.time||[]).forEach((time,i)=>{
      const el=node('div',undefined,'weather-hour');
      el.append(node('span',String(time).slice(11,16)),node('div',icon(h.weather_code?.[i]),'weather-icon'),node('b',display(h.temperature_2m?.[i],'°')),
        node('div',display(h.precipitation_probability?.[i],'%'),'muted'),node('div',display(h.precipitation?.[i],` ${hu.precipitation||'мм'}`,1),'muted'));
      if(number(h.snowfall?.[i])&&h.snowfall[i]>0)el.append(node('div',`Снег ${display(h.snowfall[i],` ${hu.snowfall||'см'}`,1)}`,'muted'));
      el.title=`${String(time).replace('T',' ')} · ${data.conditions?.hourly?.[i]||''}`;
      const bar=node('div',undefined,'weather-probability'), fill=node('span');
      fill.style.width=`${number(h.precipitation_probability?.[i])?Math.max(0,Math.min(100,h.precipitation_probability[i])):0}%`;
      bar.append(fill);el.append(bar);$('weather-hours').append(el);
    });
    $('weather-days').replaceChildren();
    (d.time||[]).forEach((time,i)=>{
      const el=node('div',undefined,'weather-day');
      el.append(node('span',`${String(time).slice(8,10)}.${String(time).slice(5,7)}`),node('div',icon(d.weather_code?.[i]),'weather-icon'),
        node('b',display(d.temperature_2m_max?.[i],'°')),node('div',display(d.temperature_2m_min?.[i],'°'),'muted'),
        node('div',display(d.precipitation_probability_max?.[i],'%'),'muted'),node('div',display(d.precipitation_sum?.[i],` ${du.precipitation_sum||'мм'}`,1),'muted'));
      el.title=data.conditions?.daily?.[i]||'';$('weather-days').append(el);
    });
    coordinates=data.coordinates;map();
  }
  async function load(region='',chosenCity=city.value,chosenCountry=country.value,locationId=''){
    if(busy)return;
    const queryCity=chosenCity.trim();
    if(queryCity.length<2){status.textContent='Укажи город.';return;}
    if(!app?.initData){status.textContent='Открой миниапп внутри Telegram.';return;}
    busy=true;$('weather-submit').disabled=true;status.className='muted';status.textContent='Получаю прогноз…';
    $('weather-candidates').replaceChildren();
    const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),16000);
    try{
      const query=new URLSearchParams({city:queryCity});
      if(chosenCountry)query.set('country_code',chosenCountry);if(region)query.set('region',region);if(locationId)query.set('location_id',locationId);
      const response=await fetch(`/api/weather?${query}`,{headers:{'X-Telegram-Init-Data':app.initData},signal:controller.signal});
      if(!response.ok)throw new Error((await response.text()).slice(0,250)||'Погода недоступна.');
      const data=await response.json();
      if(data.status==='not_found'){status.textContent='Город не найден. Проверь название и страну.';return;}
      if(data.status==='ambiguous'){
        status.textContent='Нашёл несколько городов. Выбери нужный:';
        for(const place of data.candidates||[]){
          const button=node('button',[place.name,place.admin2,place.admin1,place.country].filter(Boolean).join(', '));button.type='button';
          button.addEventListener('click',()=>load(place.admin1||'',place.name||queryCity,place.country_code||chosenCountry,String(place.id||'')));$('weather-candidates').append(button);
        }
        return;
      }
      city.value=queryCity;render(data);status.textContent='Прогноз обновлён. Время указано для выбранного города.';
    }catch(error){status.className='error';status.textContent=error.name==='AbortError'?'Источник не ответил вовремя. Попробуй ещё раз.':error.message;}
    finally{clearTimeout(timer);busy=false;$('weather-submit').disabled=false;}
  }
  $('weather-search').addEventListener('submit',event=>{event.preventDefault();load();});
  $('weather-layer').addEventListener('change',map);
  $('weather-fullscreen').addEventListener('click',async()=>{
    try{if(document.fullscreenElement)await document.exitFullscreen();else if($('weather-map-card').requestFullscreen)await $('weather-map-card').requestFullscreen();else throw new Error();}
    catch{status.textContent='Полный экран недоступен. Карту можно открыть по ссылке под ней.';}
  });
})();
