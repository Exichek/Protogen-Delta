(() => {
  'use strict';
  const date=value=>{
    if(typeof value==='number')return new Date(value);
    if(typeof value!=='string'||!value)return null;
    const text=value.length===10?`${value}T00:00:00`:value;
    const d=new Date(/[zZ]$|[+-]\d{2}:\d{2}$/.test(text)?text:`${text}+03:00`);
    return Number.isFinite(d.valueOf())?d:null;
  };
  const format=(value,options)=>{const d=date(value);return d?new Intl.DateTimeFormat('ru-RU',{timeZone:'Europe/Moscow',hourCycle:'h23',...options}).format(d):'—';};
  window.DeltaWeatherTime={
    date,
    clock:value=>format(value,{hour:'2-digit',minute:'2-digit'}),
    full:value=>`${format(value,{day:'numeric',month:'long',hour:'2-digit',minute:'2-digit'})} МСК`,
    day:value=>format(value,{weekday:'short',day:'2-digit',month:'2-digit'}),
    utcStamp:value=>{const d=date(value);return d?d.toISOString().slice(0,13).replace(/[-T]/g,''):null;},
  };
})();
