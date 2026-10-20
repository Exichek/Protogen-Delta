(() => {
  'use strict';
  const app=window.Telegram?.WebApp, button=document.getElementById('market-fullscreen');
  const label=document.getElementById('market-fullscreen-label'), note=document.getElementById('market-fullscreen-note');
  const native=Boolean(app?.initData && app.isVersionAtLeast?.('8.0') && app.requestFullscreen && app.exitFullscreen);
  let pending=null;
  function sync(){
    const active=Boolean(document.fullscreenElement || (native && app.isFullscreen));
    button.setAttribute('aria-pressed',String(active));
    label.textContent=active?'Свернуть экран':'На весь экран';
    button.title=active?'Выйти из полноэкранного режима':'Развернуть Mini App на весь экран';
  }
  function complete(){
    clearTimeout(pending);pending=null;button.disabled=false;sync();
  }
  function failed(){
    complete();
    note.textContent='Полный экран недоступен в этом окне. Попробуй развернуть окно Telegram или обновить приложение.';
  }
  button.addEventListener('click',async()=>{
    if(pending!==null)return;
    note.textContent='';
    button.disabled=true;
    try{
      if(document.fullscreenElement){await document.exitFullscreen();complete();}
      else if(native){
        pending=setTimeout(failed,5000);
        if(app.isFullscreen)app.exitFullscreen();else app.requestFullscreen();
      }else if(document.fullscreenEnabled && document.documentElement.requestFullscreen){
        await document.documentElement.requestFullscreen();complete();
      }else{app?.expand?.();failed();}
    }catch{failed();}
  });
  document.addEventListener('fullscreenchange',()=>{note.textContent='';complete();});
  document.addEventListener('fullscreenerror',failed);
  if(native){
    app.onEvent('fullscreenChanged',()=>{note.textContent='';complete();});
    app.onEvent('fullscreenFailed',failed);
    app.onEvent('activated',sync);
  }
  sync();
})();

(() => {
  'use strict';
  const q = id => document.getElementById(id), initData = window.Telegram?.WebApp?.initData || '';
  const number = value => value == null ? '—' : Number(value).toLocaleString('ru-RU', {maximumFractionDigits:6});
  const price = value => value == null ? '—' : Number(value).toLocaleString('ru-RU', {maximumFractionDigits:market()==='crypto'||Math.abs(value)<1?6:2});
  const quantity = value => value == null ? '—' : Number(value).toLocaleString('ru-RU', {maximumFractionDigits:8});
  const money = value => {const [whole,fraction]=String(value).split('.');return BigInt(whole).toLocaleString('ru-RU')+(fraction ? ','+fraction : '');};
  const date = value => {if(!value)return 'Дата не указана';const d=new Date(value);return Number.isNaN(d.valueOf())?String(value):d.toLocaleString('ru-RU');};
  const text = (id,value) => {q(id).textContent=value;};
  const periodDays={'1d':1,'1w':7,'1m':31,'3m':93,'1y':366}, intervalDays={'1m':1,'10m':7,'1h':31,'1d':366};
  let quotes=[],candles=[],selected='',sort='all',loaded=false,quoteVersion=0,chartVersion=0,bookVersion=0,chatBusy=false,interval='1h',chartZone='Europe/Moscow',point=-1,viewStart=0,viewCount=80;
  async function api(path,method='GET',body=null) {
    if(!initData)throw new Error('Открой раздел кнопкой Mini App внутри Telegram.');
    const response=await fetch('/api/markets/'+path,{method,headers:{'X-Telegram-Init-Data':initData,'Content-Type':'application/json'},body:body?JSON.stringify(body):null});
    if(!response.ok){const detail=await response.text();throw new Error(response.status===401?'Сессия истекла. Открой Mini App заново из Telegram.':detail);}
    return response.json();
  }
  function market(){return q('market-kind').value;}
  function symbolHint(){q('market-symbols').placeholder={ru:'SBER, OZON, YDEX, T',us:'SPY, QQQ, AAPL, NVDA',crypto:'BTC/USD, ETH/USD, SOL/USD'}[market()];}
  function candleDate(value){return new Date(value).toLocaleString('ru-RU',{timeZone:chartZone,day:'2-digit',month:'2-digit',year:'numeric',...(interval==='1d'?{}:{hour:'2-digit',minute:'2-digit'})});}
  function context(){text('market-chat-context',selected?`Выбрано: ${selected} · ${q('market-kind').selectedOptions[0].textContent} · свеча ${interval}`:'Можно обсудить выбранный список инструментов.');}
  function renderPrice(){
    const row=quotes.find(item=>item.symbol===selected);
    text('market-last-price',row?.price!=null?`${price(row.price)} ${row.currency==='RUB'?'₽':row.currency}`:'—');
    text('market-last-change',row?.change_pct!=null?`${row.change_pct>0?'+':''}${Number(row.change_pct).toLocaleString('ru-RU',{maximumFractionDigits:2})}%`:'');
    q('market-last-change').className=row?.change_pct>0?'market-positive':row?.change_pct<0?'market-negative':'';
    text('market-price-note',row?`${row.name} · изменение к предыдущему закрытию · время источника: ${date(row.as_of)}`:'');
  }
  function selectInstrument(symbol){selected=symbol;renderQuotes();renderPrice();context();loadChart();loadBook();}
  function renderQuotes(){
    const search=q('market-filter').value.trim().toLocaleLowerCase('ru-RU');
    const visible=quotes.filter(row=>(!search||`${row.symbol} ${row.name}`.toLocaleLowerCase('ru-RU').includes(search))&&(sort==='all'||(sort==='up'?row.change_pct>0:row.change_pct<0)));
    if(sort!=='all')visible.sort((a,b)=>sort==='up'?b.change_pct-a.change_pct:a.change_pct-b.change_pct);
    const rows=visible.map(row=>{
      const tr=document.createElement('tr'),first=document.createElement('td'),button=document.createElement('button');
      button.type='button';button.textContent=row.symbol;button.title=`${row.name}\n${row.source}\nВремя источника: ${date(row.as_of)}`;button.setAttribute('aria-pressed',String(selected===row.symbol));button.onclick=()=>selectInstrument(row.symbol);
      const name=document.createElement('small');name.className='market-muted';name.textContent=row.name===row.symbol?date(row.as_of):row.name;first.append(button,name);tr.append(first);
      for(const value of [price(row.price)+' '+row.currency,row.change_pct==null?'—':(row.change_pct>0?'+':'')+Number(row.change_pct).toLocaleString('ru-RU',{maximumFractionDigits:2})+'%',number(row.volume)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}
      tr.children[2].className=row.change_pct>0?'market-positive':row.change_pct<0?'market-negative':'';return tr;
    });
    if(!rows.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent=search?'В этом списке ничего не найдено.':sort==='all'?'Данных пока нет.':'В выбранном списке таких инструментов нет.';tr.append(td);rows.push(tr);}
    q('market-quotes').replaceChildren(...rows);
  }
  function clearChart(){++chartVersion;candles=[];point=-1;viewStart=0;viewCount=80;q('market-chart-position').hidden=true;draw();q('market-candle-rows').replaceChildren();text('market-chart-value','');text('market-chart-window','');text('market-chart-title',selected||'График');}
  function clearBook(){++bookVersion;q('market-book-refresh').disabled=false;q('market-bids').replaceChildren();q('market-asks').replaceChildren();text('market-book-spread','');text('market-book-mode','Нет данных');text('market-book-symbol',selected);text('market-book-note','');}
  async function loadQuotes(){
    const version=++quoteVersion;++chartVersion;++bookVersion;q('market-refresh').disabled=true;text('market-status','Загружаю котировки…');
    try{
      const result=await api('quotes?'+new URLSearchParams({market:market(),symbols:q('market-symbols').value}));
      if(version!==quoteVersion)return;quotes=result.quotes;
      if(!quotes.some(row=>row.symbol===selected))selected=quotes[0]?.symbol||'';
      renderQuotes();renderPrice();context();text('market-status',`Проверено: ${date(result.checked_at)} · ${quotes.length} инструментов. ${result.notices.join(' ')}`);
      text('market-quote-note',[result.ranking_scope,...new Set(quotes.map(row=>`${row.source}. ${row.notice}`))].join('\n'));
      await Promise.all([loadChart(),loadBook()]);
    }catch(error){if(version===quoteVersion){quotes=[];selected='';renderQuotes();renderPrice();context();text('market-status',error.message);clearChart();clearBook();}}
    finally{if(version===quoteVersion)q('market-refresh').disabled=false;}
  }
  function updateIntervals(changed){
    const days=periodDays[q('market-period').value];
    if(days>intervalDays[interval]){
      if(changed==='interval')q('market-period').value=Object.keys(periodDays).find(key=>periodDays[key]===intervalDays[interval]);
      else interval=Object.keys(intervalDays).find(key=>intervalDays[key]>=days);
    }
    for(const button of document.querySelectorAll('[data-market-interval]'))button.setAttribute('aria-pressed',String(button.dataset.marketInterval===interval));
    context();
  }
  async function loadChart(){
    clearChart();const version=chartVersion;
    if(!selected){text('market-chart-note','Выбери доступный инструмент.');return;}
    text('market-chart-note','Загружаю свечи…');
    try{
      const result=await api('chart?'+new URLSearchParams({market:market(),symbol:selected,period:q('market-period').value,interval}));
      if(version!==chartVersion)return;candles=result.candles;chartZone=result.timezone;viewCount=Math.min(80,candles.length);viewStart=Math.max(0,candles.length-viewCount);updateWindow();
      text('market-chart-note',`${result.source}. ${result.notice}${result.truncated?' Период получен не целиком: достигнут лимит данных.':''}${!candles.length?' Нет свечей за этот период.':''}`);
      const rows=[...candles].reverse().map(c=>{const tr=document.createElement('tr');for(const value of [candleDate(c.date),price(c.open),price(c.high),price(c.low),price(c.close),number(c.volume)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;});q('market-candle-rows').replaceChildren(...rows);
      if(candles.length)candleValue(candles[candles.length-1]);
    }catch(error){if(version===chartVersion)text('market-chart-note',error.message);}
  }
  async function loadBook(){
    clearBook();const version=bookVersion;q('market-book-refresh').disabled=true;
    if(!selected){text('market-book-note','Выбери инструмент.');q('market-book-refresh').disabled=false;return;}
    text('market-book-note','Загружаю заявки…');
    try{
      const result=await api('orderbook?'+new URLSearchParams({market:market(),symbol:selected}));
      if(version!==bookVersion)return;
      text('market-book-mode',result.kind==='depth'?'Глубина до 10 уровней':result.kind==='top'?'Только лучшие цены':'Недоступен');
      text('market-book-symbol',`${result.symbol} · ${result.source} · ${result.currency}`);
      text('market-bid-unit',result.unit);text('market-ask-unit',result.unit);
      const levels=[...result.bids,...result.asks],maxSize=Math.max(1,...levels.map(row=>row.quantity||0));
      for(const [id,items,colour] of [['market-bids',result.bids,'market-positive'],['market-asks',result.asks,'market-negative']]){
        const rows=items.map(row=>{const div=document.createElement('div');div.className='market-book-row';div.style.setProperty('--depth',`${Math.max(0,Math.min(100,(row.quantity||0)/maxSize*100))}%`);const p=document.createElement('span'),v=document.createElement('span');p.className=colour;p.textContent=price(row.price);v.textContent=quantity(row.quantity);div.append(p,v);return div;});
        if(!rows.length){const empty=document.createElement('div');empty.className='market-book-empty';empty.textContent='Нет заявок';rows.push(empty);}q(id).replaceChildren(...rows);
      }
      text('market-book-spread',result.crossed?'Цены пересекаются; спред не рассчитывается.':result.spread==null?'Спред недоступен':`Спред: ${price(result.spread)} ${result.currency} · ${Number(result.spread_pct).toLocaleString('ru-RU',{maximumFractionDigits:4})}%`);
      text('market-book-note',`${result.notice}${result.lot_size?' Размер лота: '+quantity(result.lot_size)+' '+new Intl.PluralRules('ru-RU').select(result.lot_size).replace('one','акция').replace('few','акции').replace('many','акций').replace('other','акции')+'.':''} ${result.as_of?'Время источника: '+date(result.as_of):'Время заявок не предоставлено.'} Снимок проверен: ${date(result.checked_at)}.`);
    }catch(error){if(version===bookVersion)text('market-book-note',error.message);}
    finally{if(version===bookVersion)q('market-book-refresh').disabled=false;}
  }
  const canvas=q('market-chart');
  function candleValue(c){text('market-chart-value',`${candleDate(c.date)} · Откр. ${price(c.open)} · Макс. ${price(c.high)} · Мин. ${price(c.low)} · Закр. ${price(c.close)} · Объём ${number(c.volume)}`);}
  function draw(){
    const width=canvas.clientWidth||300,height=canvas.clientHeight||320,scale=window.devicePixelRatio||1;canvas.width=width*scale;canvas.height=height*scale;const ctx=canvas.getContext('2d');if(!ctx)return;ctx.scale(scale,scale);ctx.fillStyle='#0d1118';ctx.fillRect(0,0,width,height);
    if(!candles.length){ctx.fillStyle='#93a1b5';ctx.font='14px sans-serif';ctx.fillText('Нет данных для графика',12,height/2);return;}
    const visible=candles.slice(viewStart,viewStart+viewCount);
    const left=10,right=82,top=20,bottom=height-95,volumeBase=height-28,edge=width-right,plot=Math.max(1,edge-left),min=Math.min(...visible.map(c=>c.low)),max=Math.max(...visible.map(c=>c.high)),rawSpan=max-min||max*.01||1,padding=rawSpan*.08,low=min-padding,high=max+padding,span=high-low;
    const y=value=>bottom-(value-low)/span*(bottom-top),step=plot/visible.length;
    ctx.font='11px sans-serif';ctx.lineWidth=1;ctx.strokeStyle='#ffffff0d';ctx.fillStyle='#8998ad';
    for(let i=0;i<=4;i++){const yy=top+(bottom-top)*i/4;ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(edge,yy);ctx.stroke();ctx.fillText(price(high-span*i/4),edge+7,yy+4);}
    const maxVolume=Math.max(1,...visible.map(c=>c.volume||0));
    visible.forEach((c,i)=>{const x=left+(i+.5)*step,color=c.close>=c.open?'#41d39d':'#ff727e',size=Math.max(.35,Math.min(11,step*.65));ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(x,y(c.high));ctx.lineTo(x,y(c.low));ctx.stroke();ctx.fillRect(x-size/2,Math.min(y(c.open),y(c.close)),size,Math.max(1,Math.abs(y(c.open)-y(c.close))));if(c.volume!=null){ctx.globalAlpha=.35;ctx.fillRect(x-size/2,volumeBase-c.volume/maxVolume*42,size,c.volume/maxVolume*42);ctx.globalAlpha=1;}});
    const last=visible[visible.length-1],lastY=y(last.close);ctx.setLineDash([3,4]);ctx.strokeStyle=last.close>=last.open?'#41d39d66':'#ff727e66';ctx.beginPath();ctx.moveTo(left,lastY);ctx.lineTo(edge,lastY);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=last.close>=last.open?'#174c3d':'#542b35';ctx.fillRect(edge+2,lastY-8,right-4,17);ctx.fillStyle='#edf1f7';ctx.fillText(price(last.close),edge+7,lastY+4);
    if(point>=viewStart&&point<viewStart+viewCount){const x=left+(point-viewStart+.5)*step;ctx.strokeStyle='#c3d6ef66';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,volumeBase);ctx.stroke();}
    ctx.fillStyle='#8998ad';ctx.fillText(new Date(visible[0].date).toLocaleDateString('ru-RU',{timeZone:chartZone}),left,height-10);ctx.textAlign='right';ctx.fillText(new Date(last.date).toLocaleDateString('ru-RU',{timeZone:chartZone}),edge,height-10);ctx.textAlign='left';
  }
  canvas.addEventListener('pointermove',event=>{if(!candles.length)return;const x=event.clientX-canvas.getBoundingClientRect().left;point=viewStart+Math.max(0,Math.min(viewCount-1,Math.floor((x-10)/Math.max(1,canvas.clientWidth-92)*viewCount)));candleValue(candles[point]);draw();});
  canvas.addEventListener('pointerleave',()=>{point=-1;if(candles.length)candleValue(candles[viewStart+viewCount-1]);draw();});
  canvas.addEventListener('keydown',event=>{if(!candles.length||!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();point=event.key==='Home'?0:event.key==='End'?candles.length-1:Math.max(0,Math.min(candles.length-1,(point<0?viewStart+viewCount-1:point)+(event.key==='ArrowLeft'?-1:1)));if(point<viewStart)viewStart=point;if(point>=viewStart+viewCount)viewStart=point-viewCount+1;updateWindow();candleValue(candles[point]);});
  function updateWindow(){
    if(!candles.length){draw();return;}
    const slider=q('market-chart-position');slider.max=Math.max(0,candles.length-viewCount);slider.value=viewStart;slider.hidden=Number(slider.max)===0;
    text('market-chart-window',`На экране ${viewCount} из ${candles.length} свечей · ${chartZone} · ${candleDate(candles[viewStart].date)} — ${candleDate(candles[viewStart+viewCount-1].date)}`);draw();
  }
  function zoom(count){if(!candles.length)return;const center=point>=0?point:viewStart+viewCount-1;viewCount=Math.min(candles.length,Math.max(12,count));viewStart=Math.max(0,Math.min(candles.length-viewCount,Math.round(center-viewCount/2)));point=-1;updateWindow();candleValue(candles[viewStart+viewCount-1]);}
  q('market-zoom-in').onclick=()=>zoom(Math.floor(viewCount/1.6));
  q('market-zoom-out').onclick=()=>zoom(Math.ceil(viewCount*1.6));
  q('market-zoom-all').onclick=()=>zoom(candles.length);
  q('market-chart-position').oninput=()=>{viewStart=Number(q('market-chart-position').value);point=-1;updateWindow();if(candles.length)candleValue(candles[viewStart+viewCount-1]);};
  new ResizeObserver(draw).observe(canvas);
  async function loadNews() {
    text('market-news-note','Загружаю новости…');
    try {
      const result=await api('news'), items=result.items.map(item=>{const article=document.createElement('article');article.className='market-news-item';const a=document.createElement('a');a.textContent=item.title;a.href=item.url;a.target='_blank';a.rel='noopener noreferrer';const p=document.createElement('p');p.textContent=item.summary;const meta=document.createElement('div');meta.className='market-muted';meta.textContent=`${item.kind} · ${item.source} · ${date(item.published)}`;article.append(a,p,meta);return article;});q('market-news').replaceChildren(...items);text('market-news-note',`${result.notices.join(' ')} Проверено: ${date(result.checked_at)}${!items.length?' Доступных публикаций пока нет.':''}`);
    } catch(error) {q('market-news').replaceChildren();text('market-news-note',error.message);}
  }
  async function loadRates() {
    try {
      const result=await api('rates'), popular=['RUB','KZT','USD','EUR','CNY','GBP','JPY','CHF','TRY','BYN'], currencies=result.currencies.sort((a,b)=>(popular.indexOf(a.code)<0?100:popular.indexOf(a.code))-(popular.indexOf(b.code)<0?100:popular.indexOf(b.code))||a.code.localeCompare(b.code));
      for(const id of ['currency-from','currency-to']) {const select=q(id),previous=select.value;select.replaceChildren(...currencies.map(c=>{const option=document.createElement('option');option.value=c.code;option.textContent=`${c.code} · ${c.name}`;return option;}));select.value=previous || (id==='currency-from'?'RUB':'KZT');}
      text('currency-note',`ЦБ РФ · курс действует на ${result.effective_date}. Без комиссий и банковского спреда.`);
    } catch(error) {text('currency-note',error.message);}
  }
  async function convert() {
    q('currency-convert').disabled=true;text('currency-result','Считаю…');
    try {const result=await api('convert','POST',{amount:q('currency-amount').value,from:q('currency-from').value,to:q('currency-to').value});text('currency-result',`${money(result.result)} ${result.to}`);text('currency-note',`1 ${result.from} = ${number(result.rate)} ${result.to}. ЦБ РФ · ${result.effective_date}. ${result.notice}`);}
    catch(error){text('currency-result','—');text('currency-note',error.message);}finally{q('currency-convert').disabled=false;}
  }
  function bubble(value,kind) {const node=document.createElement('div');node.className='market-bubble '+kind;node.textContent=value;q('market-chat').append(node);while(q('market-chat').children.length>12)q('market-chat').firstElementChild.remove();q('market-chat').scrollTop=q('market-chat').scrollHeight;}
  q('market-chat-form').addEventListener('submit',async event=>{
    event.preventDefault();if(chatBusy)return;const question=q('market-question').value.trim();if(!question)return;chatBusy=true;q('market-chat-send').disabled=true;q('market-chat-clear').disabled=true;const currentSelection=selected;const currentMarket=market();text('market-chat-status','Проверяю источники и готовлю ответ…');
    try {const result=await api('consult','POST',{text:question,market:currentMarket,symbols:currentSelection || q('market-symbols').value,period:q('market-period').value,interval});bubble(`${currentSelection || currentMarket}\n${question}`,'user');bubble(result.answer,'assistant');q('market-question').value='';text('market-chat-status',`${result.thinking?'Ответ с размышлениями · ':''}Данные проверены ${date(result.checked_at)}.`);}
    catch(error){text('market-chat-status',error.message);}finally{chatBusy=false;q('market-chat-send').disabled=false;q('market-chat-clear').disabled=false;}
  });
  q('market-chat-clear').onclick=async()=>{try{await api('history','DELETE');q('market-chat').replaceChildren();text('market-chat-status','История консультанта очищена.');}catch(error){text('market-chat-status',error.message);}};
  q('currency-convert').onclick=convert;q('currency-swap').onclick=()=>{const previous=q('currency-from').value;q('currency-from').value=q('currency-to').value;q('currency-to').value=previous;};
  for(const button of document.querySelectorAll('[data-market-sort]'))button.onclick=()=>{sort=button.dataset.marketSort;for(const other of document.querySelectorAll('[data-market-sort]'))other.setAttribute('aria-pressed',String(other===button));renderQuotes();};
  q('market-period').onchange=()=>{updateIntervals('period');loadChart();};
  for(const button of document.querySelectorAll('[data-market-interval]'))button.onclick=()=>{interval=button.dataset.marketInterval;updateIntervals('interval');loadChart();};
  q('market-filter').oninput=renderQuotes;
  q('market-book-refresh').onclick=loadBook;
  q('market-kind').onchange=()=>{symbolHint();q('market-symbols').value='';q('market-filter').value='';selected='';quotes=[];renderQuotes();renderPrice();clearChart();clearBook();loadQuotes();};
  q('market-symbols').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();loadQuotes();}});
  q('market-refresh').onclick=()=>{loadQuotes();loadNews();loadRates();};
  symbolHint();
  q('tab-markets').addEventListener('click',()=>{if(!loaded){loaded=true;loadQuotes();loadNews();loadRates();}requestAnimationFrame(draw);});
})();
