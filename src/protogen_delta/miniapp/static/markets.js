(() => {
  'use strict';
  const q = id => document.getElementById(id), initData = window.Telegram?.WebApp?.initData || '';
  const number = value => value == null ? '—' : Number(value).toLocaleString('ru-RU', {maximumFractionDigits:6});
  const date = value => { if (!value) return 'Дата не указана'; const d = new Date(value); return Number.isNaN(d.valueOf()) ? String(value) : d.toLocaleString('ru-RU'); };
  const text = (id,value) => {q(id).textContent=value;};
  let quotes=[], candles=[], selected='', sort='all', loaded=false, quoteVersion=0, chartVersion=0, chatBusy=false;
  async function api(path, method='GET', body=null) {
    if (!initData) throw new Error('Открой раздел кнопкой Mini App внутри Telegram.');
    const response=await fetch('/api/markets/'+path,{method,headers:{'X-Telegram-Init-Data':initData,'Content-Type':'application/json'},body:body?JSON.stringify(body):null});
    if(!response.ok) { const detail=await response.text(); throw new Error(response.status===401?'Сессия истекла. Открой Mini App заново из Telegram.':detail); }
    return response.json();
  }
  function market() {return q('market-kind').value;}
  function context() {text('market-chat-context', selected ? `Выбрано: ${selected} · ${q('market-kind').selectedOptions[0].textContent}` : 'Можно обсудить выбранный список инструментов.');}
  function renderQuotes() {
    const visible=quotes.filter(row=>sort==='all'||(sort==='up'?row.change_pct>0:row.change_pct<0));
    if(sort!=='all') visible.sort((a,b)=>sort==='up'?b.change_pct-a.change_pct:a.change_pct-b.change_pct);
    const rows=visible.map(row=>{
      const tr=document.createElement('tr'), first=document.createElement('td'), button=document.createElement('button');
      button.type='button';button.textContent=row.symbol;button.title=`${row.name}\n${row.source}\nВремя сделки: ${date(row.as_of)}`;button.setAttribute('aria-pressed',String(selected===row.symbol));
      button.onclick=()=>{selected=row.symbol;renderQuotes();context();loadChart();};const stamp=document.createElement('small');stamp.className='market-muted';stamp.textContent=date(row.as_of);first.append(button,stamp);tr.append(first);
      for(const value of [number(row.price)+' '+row.currency,row.change_pct==null?'—':(row.change_pct>0?'+':'')+number(row.change_pct)+'%',number(row.volume)]) {const td=document.createElement('td');td.textContent=value;tr.append(td);}
      tr.children[2].className=row.change_pct>0?'market-positive':row.change_pct<0?'market-negative':'';return tr;
    });
    if(!rows.length) {const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=4;td.textContent=sort==='all'?'Данных пока нет.':'В выбранном списке таких инструментов нет.';tr.append(td);rows.push(tr);}
    q('market-quotes').replaceChildren(...rows);
  }
  async function loadQuotes() {
    const version=++quoteVersion;q('market-refresh').disabled=true;text('market-status','Загружаю котировки…');
    try {
      const result=await api('quotes?'+new URLSearchParams({market:market(),symbols:q('market-symbols').value}));
      if(version!==quoteVersion)return;quotes=result.quotes;
      if(!quotes.some(row=>row.symbol===selected)) selected=quotes[0]?.symbol || '';
      renderQuotes();context();text('market-status',`Проверено: ${date(result.checked_at)}. ${result.notices.join(' ')}`);
      text('market-quote-note',[result.ranking_scope,...new Set(quotes.map(row=>`${row.source}. ${row.notice}`))].join('\n'));
      await loadChart();
    } catch(error) { if(version===quoteVersion) {quotes=[];selected='';renderQuotes();context();text('market-status',error.message);await loadChart();} }
    finally {if(version===quoteVersion) q('market-refresh').disabled=false;}
  }
  async function loadChart() {
    const version=++chartVersion;candles=[];draw();q('market-candle-rows').replaceChildren();text('market-chart-title',selected || 'График');text('market-chart-value','');text('market-chart-note','');
    if(!selected) {text('market-chart-note','Выбери доступный инструмент.');return;}
    text('market-chart-note','Загружаю дневные свечи…');
    try {
      const result=await api('chart?'+new URLSearchParams({market:market(),symbol:selected,period:q('market-period').value}));
      if(version!==chartVersion)return;candles=result.candles;draw();
      text('market-chart-note',`${result.source}. ${result.notice}${result.truncated?' Показана часть периода.':''}${!candles.length?' Нет свечей за этот период.':''}`);
      const rows=[...candles].reverse().map(c=>{const tr=document.createElement('tr');for(const value of [date(c.date),number(c.open),number(c.high),number(c.low),number(c.close),number(c.volume)]) {const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;});q('market-candle-rows').replaceChildren(...rows);
      if(candles.length) candleValue(candles[candles.length-1]);
    } catch(error) {if(version===chartVersion) text('market-chart-note',error.message);}
  }
  const canvas=q('market-chart');
  function candleValue(c) {text('market-chart-value',`${date(c.date)} · Откр. ${number(c.open)} · Макс. ${number(c.high)} · Мин. ${number(c.low)} · Закр. ${number(c.close)} · Объём ${number(c.volume)}`);}
  function draw() {
    const width=canvas.clientWidth || 300,height=270,scale=window.devicePixelRatio || 1;canvas.width=width*scale;canvas.height=height*scale;const ctx=canvas.getContext('2d');ctx.scale(scale,scale);ctx.clearRect(0,0,width,height);
    if(!candles.length){ctx.fillStyle='#9cabc0';ctx.font='14px sans-serif';ctx.fillText('Нет данных для графика',12,135);return;}
    const left=4,right=62,top=16,bottom=190,plot=width-left-right,low=Math.min(...candles.map(c=>c.low)),high=Math.max(...candles.map(c=>c.high)),span=high-low || high*.01 || 1;
    const y=value=>bottom-(value-low)/span*(bottom-top), step=plot/candles.length;
    ctx.font='11px sans-serif';ctx.strokeStyle='#ffffff16';ctx.fillStyle='#a8b5c7';
    for(let i=0;i<=4;i++){const yy=top+(bottom-top)*i/4;ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(plot,yy);ctx.stroke();ctx.fillText(number(high-span*i/4),plot+6,yy+4);}
    const maxVolume=Math.max(1,...candles.map(c=>c.volume || 0));
    candles.forEach((c,i)=>{const x=left+(i+.5)*step,color=c.close>=c.open?'#64dca9':'#ff8994';ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(x,y(c.high));ctx.lineTo(x,y(c.low));ctx.stroke();const size=Math.max(1,Math.min(10,step*.65));ctx.fillRect(x-size/2,Math.min(y(c.open),y(c.close)),size,Math.max(1,Math.abs(y(c.open)-y(c.close))));ctx.globalAlpha=.4;ctx.fillRect(x-size/2,246-(c.volume||0)/maxVolume*38,size,(c.volume||0)/maxVolume*38);ctx.globalAlpha=1;});
    ctx.fillStyle='#a8b5c7';ctx.fillText(new Date(candles[0].date).toLocaleDateString('ru-RU'),4,263);ctx.fillText(new Date(candles[candles.length-1].date).toLocaleDateString('ru-RU'),Math.max(4,plot-65),263);
  }
  canvas.addEventListener('pointermove',event=>{if(!candles.length)return;const x=event.clientX-canvas.getBoundingClientRect().left;const i=Math.max(0,Math.min(candles.length-1,Math.floor((x-4)/(canvas.clientWidth-66)*candles.length)));candleValue(candles[i]);});
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
    try {const result=await api('convert','POST',{amount:q('currency-amount').value,from:q('currency-from').value,to:q('currency-to').value});text('currency-result',`${result.result} ${result.to}`);text('currency-note',`1 ${result.from} = ${number(result.rate)} ${result.to}. ЦБ РФ · ${result.effective_date}. ${result.notice}`);}
    catch(error){text('currency-result','—');text('currency-note',error.message);}finally{q('currency-convert').disabled=false;}
  }
  function bubble(value,kind) {const node=document.createElement('div');node.className='market-bubble '+kind;node.textContent=value;q('market-chat').append(node);while(q('market-chat').children.length>12)q('market-chat').firstElementChild.remove();q('market-chat').scrollTop=q('market-chat').scrollHeight;}
  q('market-chat-form').addEventListener('submit',async event=>{
    event.preventDefault();if(chatBusy)return;const question=q('market-question').value.trim();if(!question)return;chatBusy=true;q('market-chat-send').disabled=true;q('market-chat-clear').disabled=true;const currentSelection=selected;const currentMarket=market();text('market-chat-status','Проверяю источники и готовлю ответ…');
    try {const result=await api('consult','POST',{text:question,market:currentMarket,symbols:currentSelection || q('market-symbols').value,period:q('market-period').value});bubble(`${currentSelection || currentMarket}\n${question}`,'user');bubble(result.answer,'assistant');q('market-question').value='';text('market-chat-status',`${result.thinking?'Ответ с размышлениями · ':''}Данные проверены ${date(result.checked_at)}.`);}
    catch(error){text('market-chat-status',error.message);}finally{chatBusy=false;q('market-chat-send').disabled=false;q('market-chat-clear').disabled=false;}
  });
  q('market-chat-clear').onclick=async()=>{try{await api('history','DELETE');q('market-chat').replaceChildren();text('market-chat-status','История консультанта очищена.');}catch(error){text('market-chat-status',error.message);}};
  q('currency-convert').onclick=convert;q('currency-swap').onclick=()=>{const previous=q('currency-from').value;q('currency-from').value=q('currency-to').value;q('currency-to').value=previous;};
  for(const button of document.querySelectorAll('[data-market-sort]'))button.onclick=()=>{sort=button.dataset.marketSort;for(const other of document.querySelectorAll('[data-market-sort]'))other.setAttribute('aria-pressed',String(other===button));renderQuotes();};
  q('market-period').onchange=loadChart;
  q('market-kind').onchange=()=>{q('market-symbols').value='';selected='';loadQuotes();};
  q('market-symbols').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();loadQuotes();}});
  q('market-refresh').onclick=()=>{loadQuotes();loadNews();loadRates();};
  q('tab-markets').addEventListener('click',()=>{if(!loaded){loaded=true;loadQuotes();loadNews();loadRates();}requestAnimationFrame(draw);});
})();
