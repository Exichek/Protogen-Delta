const tg = window.Telegram?.WebApp; if (tg) { tg.ready(); tg.expand(); }
  const initData = tg?.initData || ""; const status = document.querySelector("#status");
  const api = async (method="GET", body=null, path="/api/profile", extraHeaders={}) => {
    const file = body instanceof File;
    const response = await fetch(path, {method, headers:{"X-Telegram-Init-Data":initData,"Content-Type":file ? (body.type || "application/octet-stream") : "application/json",...extraHeaders}, body:file ? body : body?JSON.stringify(body):null});
    if (!response.ok) {
      const message = await response.text();
      if (response.status === 401) throw new Error(message === "initData просрочен"
        ? "Срок доступа к панели истёк. Полностью закрой Mini App и открой заново через «RP и функции»."
        : "Не удалось подтвердить доступ. Полностью закрой панель и открой заново кнопкой бота внутри Telegram.");
      throw new Error(message || `HTTP ${response.status}`);
    }
    return response.json();
  };
  const render = p => {
    document.querySelector("#hello").textContent = `${p.user.first_name} · ID ${p.user.id}`;
    document.querySelector("#content-mode").value = p.content_mode === "adult" ? "adult" : "soft";
    document.querySelector('#content-mode option[value="adult"]').disabled = Boolean(p.age_restricted);
    document.querySelector("#age-restriction-hint").hidden = !p.age_restricted;
    document.querySelector("#stopword").value = p.roleplay_stopword || "стоп";
    document.querySelector("#configuration").value = p.roleplay_configuration;
    document.querySelector("#rp-allow-profanity").checked = Boolean(p.roleplay_allow_profanity);
    renderCharacter(p);
    document.querySelector("#preferences").value = p.roleplay_preferences;
    document.querySelector("#boundaries").value = p.roleplay_boundaries;
    document.querySelector("#fetishes").textContent = `Распознанные темы: ${p.roleplay_fetishes.length ? p.roleplay_fetishes.join(", ") : "пока нет"}`;
    document.querySelector("#mode").textContent = p.content_mode === "adult" ? "18+" : "Мягкий";
    document.querySelector("#rp-state").textContent = p.roleplay_active ? "Активна" : "Выключена";
    renderAppearance(p, true);
  };
  const archiveFile = document.querySelector('#archive-file');
  const archiveView = document.querySelector('#archive-view');
  const archiveAnalyze = document.querySelector('#archive-analyze');
  const archiveStatus = document.querySelector('#archive-status');
  const archiveResult = document.querySelector('#archive-result');
  let archiveBusy = false;
  const archiveUpdate = () => {
    archiveView.disabled = archiveAnalyze.disabled = archiveBusy || !archiveFile.files?.length || !initData;
  };
  archiveFile.addEventListener('change', () => {
    archiveResult.textContent = ''; archiveResult.hidden = true; archiveStatus.textContent = ''; archiveUpdate();
  });
  const inspectArchive = async analyze => {
    const file = archiveFile.files?.[0];
    if (!file || archiveBusy) return;
    if (!file.name.toLowerCase().endsWith('.zip') || file.size > 50 * 1024 * 1024) {
      archiveStatus.textContent = 'Выбери ZIP до 50 МБ.'; return;
    }
    archiveBusy = true; archiveUpdate(); archiveFile.disabled = true;
    archiveResult.hidden = true;
    archiveStatus.textContent = analyze ? 'Читаю файлы и готовлю ответ Дельты…' : 'Читаю файлы архива…';
    try {
      const options = encodeURIComponent(JSON.stringify({file_name:file.name,
        question:document.querySelector('#archive-question').value, analyze}));
      if (options.length > 7500) throw new Error('Сократи вопрос или название архива.');
      const result = await api('POST', file, '/api/tools/archive', {
        'X-Archive-Options': options
      });
      archiveResult.textContent = (result.reply ? `Ответ Дельты:\n${result.reply}\n\n` : '') + result.text;
      archiveResult.hidden = false;
      archiveStatus.textContent = result.truncated ? 'Просмотр частичный: достигнут лимит объёма, числа файлов или времени.' : 'Готово.';
    } catch (error) { archiveStatus.textContent = error.message; }
    finally { archiveBusy = false; archiveFile.disabled = false; archiveUpdate(); }
  };
  archiveView.addEventListener('click', () => inspectArchive(false));
  archiveAnalyze.addEventListener('click', () => inspectArchive(true));
  const appearanceStatus = document.querySelector("#appearance-status");
  const fileInput = document.querySelector("#appearance-file");
  const preview = document.querySelector("#appearance-preview");
  const appearanceMode = document.querySelector("#appearance-mode");
  const descriptionInput = document.querySelector("#appearance-description");
  const appearanceInputArea = document.querySelector("#appearance-input-area");
  const pasteAppearance = document.querySelector("#paste-appearance");
  const characterInputArea = document.querySelector("#character-input-area");
  const characterUpload = document.querySelector("#character-upload");
  const characterFileInput = document.querySelector("#character-file");
  const characterPreview = document.querySelector("#character-preview");
  const imageTypes = ["image/png", "image/jpeg", "image/webp"];
  const appearanceEditor = document.querySelector("#appearance-editor");
  const editText = document.querySelector("#appearance-edit-text");
  let appearanceEditing = false, savedDescription = "", editOriginal = "", savedSpecies = null;
  const profileFields = ["personality", "behavior", "preferences", "fetishes", "boundaries"];
  const emptyProfile = () => Object.fromEntries(profileFields.map(name => [name, ""]));
  const profileDraft = () => Object.fromEntries(profileFields.map(name => [name, document.querySelector(`#delta-${name}`).value.trim()]));
  let savedProfile = emptyProfile(), characterBusy = false, settingsBusy = false, characterSnapshot = "";
  let savedTopicText = "Распознанные темы: пока нет.";
  const updateAppearanceTopicHint = () => {
    document.querySelector("#delta-fetish-topics").textContent = document.querySelector("#delta-fetishes").value.trim() === savedProfile.fetishes ? savedTopicText : "Распознанные темы обновятся после сохранения.";
  };
  let appearanceBusy = false, clipboardBusy = false, appearanceEnabled = false, customAppearance = false, selectedFile = null, previewUrl = null, selectionRevision = 0;
  let selectedCharacterFile = null, characterPreviewUrl = null, characterSelectionRevision = 0, imagePasteTarget = "delta";
  const updateAppearanceButtons = () => {
    const busy = appearanceBusy || clipboardBusy || appearanceEditing || characterBusy || settingsBusy;
    document.querySelector("#edit-appearance").hidden = !customAppearance || appearanceEditing;
    document.querySelector("#edit-appearance").disabled = busy || !appearanceEnabled;
    document.querySelector("#save-appearance-edit").disabled = appearanceBusy || !editText.value.trim() || editText.value.trim() === editOriginal;
    document.querySelector("#cancel-appearance-edit").disabled = appearanceBusy;
    editText.disabled = appearanceBusy;
    document.querySelector("#choose-appearance").disabled = busy || !appearanceEnabled;
    document.querySelector("#apply-appearance").disabled = busy || !appearanceEnabled || !(selectedFile || (appearanceMode.value === "text" && descriptionInput.value.trim()));
    document.querySelector("#reset-appearance").disabled = busy || !customAppearance;
    pasteAppearance.hidden = appearanceMode.value !== "image";
    pasteAppearance.disabled = busy || !appearanceEnabled;
    appearanceInputArea.setAttribute("aria-busy", String(busy));
    fileInput.disabled = busy || !appearanceEnabled;
    appearanceMode.disabled = busy;
    descriptionInput.disabled = busy || !appearanceEnabled;
    document.querySelector("#appearance-species").disabled = busy || !appearanceEnabled;
    const speciesValue = document.querySelector("#appearance-species").value.trim();
    document.querySelector("#save-appearance-species").disabled = busy || !appearanceEnabled || !customAppearance || Boolean(selectedFile) || !speciesValue || (savedSpecies?.source === "user" && speciesValue === savedSpecies.name);
    document.querySelector("#appearance-notes").disabled = busy || !appearanceEnabled;
    document.querySelector("#appearance-same-character").disabled = busy || !appearanceEnabled || !customAppearance;
    for (const name of profileFields) document.querySelector(`#delta-${name}`).disabled = busy || !appearanceEnabled;
    updateAppearanceTopicHint();
    document.querySelector("#save-delta-profile").disabled = busy || !appearanceEnabled || Boolean(selectedFile) || JSON.stringify(profileDraft()) === JSON.stringify(savedProfile);
    updateCharacterButtons();
  };
  const renderAppearance = (p, preserveProfileDraft=false) => {
    const nextDescription = p.delta_appearance || "";
    const nextProfile = {...emptyProfile(), ...(p.delta_appearance_profile || {})};
    const topics = (p.delta_appearance_topics || []).map(topic => topic.name);
    savedTopicText = topics.length ? `Распознанные темы: ${topics.join(", ")}.` : "Распознанные темы: пока нет.";
    const keepDraft = preserveProfileDraft && nextDescription === savedDescription && JSON.stringify(nextProfile) === JSON.stringify(savedProfile);
    savedDescription = nextDescription; savedProfile = nextProfile;
    if (!keepDraft) {
      for (const name of profileFields) document.querySelector(`#delta-${name}`).value = savedProfile[name];
      document.querySelector("#delta-profile-status").textContent="";
    }
    document.querySelector("#delta-profile-hint").textContent = savedDescription ? "Для текущего облика. Заполни вручную: эти сведения не определяются по картинке." : "Для стандартного Дельты. Настройки этого образа сохраняются отдельно от загруженного облика.";
    customAppearance = Boolean(p.delta_appearance);
    appearanceEnabled = Boolean(p.appearance_upload_enabled && initData);
    const species = p.delta_species;
    savedSpecies = species || null;
    // Сохранённый текст нужен целиком для редактирования и проверки версии.
    // При отображении убираем только заголовок, добавленный приложением.
    let visibleDescription = savedDescription;
    if (species && typeof species.name === "string") {
      const header = species.source === "user" ? `Вид со слов пользователя: ${species.name}. ` : species.status === "probable" ? `Вероятно, ${species.name}. ` : `${species.name}. `;
      if (visibleDescription.startsWith(header)) visibleDescription = visibleDescription.slice(header.length);
    }
    document.querySelector("#appearance").textContent = visibleDescription || (customAppearance ? savedDescription : "Базовый облик");
    document.querySelector("#appearance-empty-hint").hidden = customAppearance;
    if (p.delta_reference_restricted) appearanceStatus.textContent = "Этот референс используется только для нейтрального общения.";
    const thumbnail = p.delta_appearance && p.delta_appearance_thumbnail;
    const savedImage = document.querySelector("#saved-appearance-image");
    // Сервер возвращает только ограниченные JPEG data URL, внешние адреса не загружаем.
    const hasThumbnail = typeof thumbnail === "string" && thumbnail.length <= 90000 && /^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(thumbnail);
    if (hasThumbnail) savedImage.src = thumbnail; else savedImage.removeAttribute("src");
    document.querySelector("#saved-appearance").hidden = !hasThumbnail;
    const speciesStatus = document.querySelector("#appearance-species-status");
    speciesStatus.hidden = !customAppearance || !species;
    speciesStatus.textContent = !species ? "" : species.status === "unknown" ? "Вид пока не определён" : species.source === "user" ? `Вид: ${species.name} · указан тобой` : species.status === "hybrid" ? "Смешанный или авторский дизайн · по изображению" : `Вид: ${species.name} · по изображению`;
    const checkLabels = {match:"Видимые признаки согласуются с указанным видом.",unconfirmed:"По этому ракурсу указанный вид пока не подтверждён.",conflict:"Распознавание расходится с указанным видом. Проверь описание или добавь более ясный референс."};
    const speciesCheck = document.querySelector("#appearance-species-check");
    const checkLabel = species && species.source === "user" ? checkLabels[species.visual_check] || "" : "";
    speciesCheck.textContent = checkLabel;
    speciesCheck.hidden = !customAppearance || !checkLabel;
    updateAppearanceButtons();
  };
  const updateCharacterButtons = () => {
    const busy = characterBusy || appearanceBusy || clipboardBusy || appearanceEditing || settingsBusy;
    characterFileInput.disabled = busy || !appearanceEnabled;
    document.querySelector("#choose-character").disabled = busy || !appearanceEnabled;
    document.querySelector("#paste-character").disabled = busy || !appearanceEnabled;
    characterInputArea.setAttribute("aria-busy", String(busy));
    document.querySelector("#character-file-name").textContent = selectedCharacterFile?.name || "";
    document.querySelector("#analyze-character").disabled = busy || !appearanceEnabled || !selectedCharacterFile;
    document.querySelector("#save-character-description").disabled = busy || document.querySelector("#character").value.trim() === characterSnapshot;
    for (const id of ["character", "character-species", "character-notes"]) document.querySelector(`#${id}`).disabled = busy;
    document.querySelector("#save").disabled = busy;
    document.querySelector("#rp-allow-profanity").disabled = busy;
  };
  const renderCharacter = p => {
    characterSnapshot = p.roleplay_character || "";
    document.querySelector("#rp-state").textContent = p.roleplay_active ? "Активна" : "Выключена";
    document.querySelector("#character").value = characterSnapshot;
    const thumbnail = characterSnapshot && p.roleplay_character_thumbnail;
    const hasThumbnail = typeof thumbnail === "string" && thumbnail.length <= 90000 && /^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(thumbnail);
    const image = document.querySelector("#saved-character-image");
    if (hasThumbnail) image.src = thumbnail; else image.removeAttribute("src");
    document.querySelector("#saved-character").hidden = !hasThumbnail;
    updateCharacterButtons();
  };
  const characterStatus = document.querySelector("#character-status");
  document.querySelector("#character").addEventListener("input", () => { updateCharacterButtons(); characterStatus.className=""; characterStatus.textContent=document.querySelector("#character").value.trim() === characterSnapshot ? "" : "Есть несохранённые изменения."; });
  const characterFail = error => { characterStatus.className="error"; characterStatus.textContent=error.message; };
  const clearCharacterSelection = () => {
    characterSelectionRevision++;
    if (characterPreviewUrl) URL.revokeObjectURL(characterPreviewUrl);
    characterPreviewUrl=null; selectedCharacterFile=null; characterFileInput.value="";
    characterPreview.removeAttribute("src"); characterPreview.hidden=true;
  };
  const imageFileError = file => file.size > 20 * 1024 * 1024 ? "Картинка должна быть до 20 МБ." : !imageTypes.includes(file.type) && !(file.type === "" && /\.(jpe?g|png|webp)$/i.test(file.name)) ? "Выбери картинку JPEG, PNG или WebP." : "";
  const selectCharacterFile = file => {
    if (!file || characterBusy || appearanceBusy || appearanceEditing || settingsBusy || !appearanceEnabled) return false;
    const error=imageFileError(file);
    if (error) { characterFail(new Error(error)); return false; }
    clearCharacterSelection(); selectedCharacterFile=file; characterPreviewUrl=URL.createObjectURL(file);
    characterPreview.src=characterPreviewUrl; characterPreview.hidden=false; imagePasteTarget="character";
    characterStatus.className=""; characterStatus.textContent=""; updateAppearanceButtons(); return true;
  };
  document.querySelector("#choose-character").addEventListener("click", () => characterFileInput.click());
  characterFileInput.addEventListener("change", () => selectCharacterFile(characterFileInput.files[0]));
  characterUpload.addEventListener("toggle", () => {
    if (characterUpload.open) imagePasteTarget="character";
    else { characterSelectionRevision++; if (imagePasteTarget === "character") imagePasteTarget="delta"; }
  });
  document.querySelector("#save-character-description").addEventListener("click", async () => {
    if (document.querySelector("#save-character-description").disabled) return;
    characterBusy = true; updateAppearanceButtons(); characterStatus.className=""; characterStatus.textContent="Сохраняю описание…";
    try {
      const profile = await api("PATCH", {roleplay_character:document.querySelector("#character").value,expected_character:characterSnapshot});
      renderCharacter(profile); characterStatus.className="ok"; characterStatus.textContent="Описание твоего персонажа сохранено.";
    } catch(error) { characterStatus.className="error"; characterStatus.textContent=error.message; }
    finally { characterBusy=false; updateAppearanceButtons(); }
  });
  document.querySelector("#analyze-character").addEventListener("click", async () => {
    if (document.querySelector("#analyze-character").disabled) return;
    const file = selectedCharacterFile;
    characterBusy=true; updateAppearanceButtons(); characterStatus.className=""; characterStatus.textContent="Разбираю твоего персонажа…";
    try {
      const headers = {"X-Appearance-Options":encodeURIComponent(JSON.stringify({species:document.querySelector("#character-species").value.trim(),notes:document.querySelector("#character-notes").value.trim()}))};
      const profile=await api("POST",file,"/api/profile/character/appearance",headers);
      renderCharacter(profile); clearCharacterSelection(); document.querySelector("#character-species").value=""; document.querySelector("#character-notes").value="";
      characterStatus.className="ok"; characterStatus.textContent="Твой облик сохранён. Проверь описание и исправь детали при необходимости.";
    } catch(error) { characterStatus.className="error"; characterStatus.textContent=error.message; }
    finally { characterBusy=false; updateAppearanceButtons(); }
  });
  for (const name of profileFields) document.querySelector(`#delta-${name}`).addEventListener("input", () => { updateAppearanceButtons(); const output=document.querySelector("#delta-profile-status"); output.className=""; output.textContent=JSON.stringify(profileDraft()) === JSON.stringify(savedProfile) ? "" : "Есть несохранённые изменения."; });
  document.querySelector("#save-delta-profile").addEventListener("click", async () => {
    if (document.querySelector("#save-delta-profile").disabled) return;
    const output=document.querySelector("#delta-profile-status"); appearanceBusy=true; updateAppearanceButtons(); output.className=""; output.textContent="Сохраняю профиль…";
    try {
      const profile=await api("PATCH",{profile:profileDraft(),expected_appearance:savedDescription,expected_profile:savedProfile},"/api/profile/appearance/behavior");
      renderAppearance(profile); output.className="ok"; output.textContent="Характер и поведение Дельты сохранены.";
    } catch(error) { output.className="error"; output.textContent=error.message; }
    finally { appearanceBusy=false; updateAppearanceButtons(); }
  });
  const clearSelection = () => {
    selectionRevision++;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; selectedFile = null; fileInput.value = "";
    preview.removeAttribute("src"); preview.hidden = true;
    document.querySelector("#appearance-file-name").textContent = "";
  };
  const appearanceFail = error => { appearanceStatus.className="error"; appearanceStatus.textContent = initData ? `Ошибка: ${error.message}` : "Открой панель кнопкой внутри Telegram."; };
  document.querySelector("#appearance-species").addEventListener("input", updateAppearanceButtons);
  document.querySelector("#save-appearance-species").addEventListener("click", async () => {
    if (document.querySelector("#save-appearance-species").disabled) return;
    const name = document.querySelector("#appearance-species").value.trim();
    appearanceBusy = true; updateAppearanceButtons(); appearanceStatus.className=""; appearanceStatus.textContent="Сохраняю вид…";
    try {
      const profile = await api("PATCH", {species:name,expected_appearance:savedDescription,expected_species:savedSpecies}, "/api/profile/appearance/species");
      renderAppearance(profile); appearanceStatus.className="ok"; appearanceStatus.textContent="Вид сохранён. Это твоё уточнение.";
    } catch(error) { appearanceFail(error); }
    finally { appearanceBusy=false; updateAppearanceButtons(); }
  });
  document.querySelector("#edit-appearance").addEventListener("click", () => {
    if (appearanceBusy || clipboardBusy || !customAppearance || !appearanceEnabled) return;
    editOriginal = savedDescription; editText.value = savedDescription;
    appearanceEditing = true; appearanceEditor.hidden = false;
    appearanceStatus.textContent = ""; updateAppearanceButtons(); editText.focus();
  });
  editText.addEventListener("input", updateAppearanceButtons);
  document.querySelector("#cancel-appearance-edit").addEventListener("click", () => {
    if (appearanceBusy) return;
    appearanceEditing = false; appearanceEditor.hidden = true; editText.value = "";
    appearanceStatus.textContent = ""; updateAppearanceButtons();
  });
  document.querySelector("#save-appearance-edit").addEventListener("click", async () => {
    if (appearanceBusy || !appearanceEditing || !editText.value.trim()) return;
    appearanceBusy = true; updateAppearanceButtons();
    appearanceStatus.className = ""; appearanceStatus.textContent = "Сохраняю исправления…";
    try {
      const profile = await api("PATCH", {appearance:editText.value, expected_appearance:editOriginal}, "/api/profile/appearance");
      appearanceEditing = false; appearanceEditor.hidden = true; editText.value = "";
      renderAppearance(profile); appearanceStatus.className = "ok";
      appearanceStatus.textContent = "Описание исправлено. Дельта будет учитывать изменения.";
    } catch (error) { appearanceFail(error); }
    finally { appearanceBusy = false; updateAppearanceButtons(); }
  });
  document.querySelector("#choose-appearance").addEventListener("click", () => fileInput.click());
  appearanceMode.addEventListener("change", () => {
    clearSelection(); appearanceStatus.textContent = "";
    const text = appearanceMode.value === "text";
    document.querySelector("#appearance-text-panel").hidden = !text;
    document.querySelector("#appearance-image-hint").hidden = text;
    document.querySelector("#appearance-species-panel").hidden = text;
    document.querySelector("#appearance-clipboard-hint").hidden = text;
    appearanceInputArea.querySelector(".appearance-actions").classList.toggle("text-source", text);
    document.querySelector("#choose-appearance").textContent = text ? "Загрузить TXT/JSON" : "Выбрать картинку";
    fileInput.accept = text ? ".txt,.json,text/plain,application/json" : "image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp";
    updateAppearanceButtons();
  });
  descriptionInput.addEventListener("input", () => { if (selectedFile) clearSelection(); updateAppearanceButtons(); });
  const selectAppearanceFile = file => {
    if (!file || appearanceBusy || characterBusy || appearanceEditing || settingsBusy || !appearanceEnabled) return false;
    if (appearanceMode.value === "text") {
      if (!/\.(txt|json)$/i.test(file.name) || file.size > 64 * 1024) { appearanceFail(new Error("Выбери TXT/JSON до 64 КБ.")); return false; }
    } else {
      const error=imageFileError(file);
      if (error) { appearanceFail(new Error(error)); return false; }
    }
    clearSelection(); imagePasteTarget="delta"; appearanceStatus.className=""; appearanceStatus.textContent="";
    if (appearanceMode.value === "text") {
      selectedFile = new File([file], file.name, {type:/\.json$/i.test(file.name) ? "application/json" : "text/plain"});
    } else {
      selectedFile = file; previewUrl = URL.createObjectURL(file); preview.src = previewUrl; preview.hidden = false;
    }
    document.querySelector("#appearance-file-name").textContent = file.name;
    updateAppearanceButtons();
    return true;
  };
  fileInput.addEventListener("change", () => selectAppearanceFile(fileInput.files[0]));
  const imageInputs = {
    delta:{area:appearanceInputArea,button:pasteAppearance,status:appearanceStatus,available:() => appearanceMode.value === "image",revision:() => selectionRevision,select:selectAppearanceFile,fail:appearanceFail,success:"Картинка добавлена. Проверь превью и нажми «Применить облик»."},
    character:{area:characterInputArea,button:document.querySelector("#paste-character"),status:characterStatus,available:() => characterUpload.open,revision:() => characterSelectionRevision,select:selectCharacterFile,fail:characterFail,success:"Картинка добавлена. Проверь превью и нажми «Распознать мой облик»."}
  };
  const inputAvailable = input => appearanceEnabled && !document.querySelector("#rp-panel").hidden && input.available();
  const canPasteImage = input => inputAvailable(input) && !appearanceBusy && !characterBusy && !clipboardBusy && !appearanceEditing && !settingsBusy;
  const pasteImageFile = (input, file) => {
    if (input.select(file)) { input.status.className="ok"; input.status.textContent=input.success; }
  };
  const readImageClipboard = async input => {
    if (!canPasteImage(input)) return;
    const fallback = "Этот клиент не разрешил прочитать буфер. Нажми Ctrl+V в блоке картинки или выбери файл.";
    if (!navigator.clipboard?.read) { input.fail(new Error(fallback)); input.area.focus({preventScroll:true}); return; }
    const revision = input.revision();
    clipboardBusy=true; updateAppearanceButtons(); input.status.className=""; input.status.textContent="Читаю картинку из буфера…";
    try {
      const items = await navigator.clipboard.read();
      let image = null;
      for (const item of items) {
        const type = imageTypes.find(type => item.types.includes(type));
        if (type) { image = {blob:await item.getType(type),type}; break; }
      }
      if (revision !== input.revision() || !inputAvailable(input)) {
        if (revision === input.revision() || !inputAvailable(input)) { input.status.className=""; input.status.textContent="Вставка отменена. Вернись к облику и повтори её."; }
        return;
      }
      if (!image) throw new Error("В буфере нет картинки JPEG, PNG или WebP. Скопируй само изображение, а не ссылку на него.");
      const extension = {"image/png":"png","image/jpeg":"jpg","image/webp":"webp"}[image.type];
      pasteImageFile(input, new File([image.blob], `Картинка из буфера.${extension}`, {type:image.type}));
    } catch(error) {
      if (!inputAvailable(input)) {
        input.status.className=""; input.status.textContent="Вставка отменена. Вернись к облику и повтори её.";
      } else if (revision === input.revision()) {
        input.fail(new Error(error.name === "NotAllowedError" || error.name === "SecurityError" ? fallback : error.message));
        input.area.focus({preventScroll:true});
      }
    } finally { clipboardBusy=false; updateAppearanceButtons(); }
  };
  for (const [name, input] of Object.entries(imageInputs)) {
    for (const event of ["focusin", "pointerdown"]) input.area.addEventListener(event, () => { imagePasteTarget=name; });
    input.button.addEventListener("click", () => readImageClipboard(input));
    input.area.addEventListener("dragover", event => {
      if (Array.from(event.dataTransfer?.types || []).includes("Files")) {
        event.preventDefault();
        if (canPasteImage(input)) input.area.classList.add("drag-over");
      }
    });
    input.area.addEventListener("dragleave", event => {
      if (!input.area.contains(event.relatedTarget)) input.area.classList.remove("drag-over");
    });
    input.area.addEventListener("drop", event => {
      input.area.classList.remove("drag-over");
      const files=Array.from(event.dataTransfer?.files || []);
      if (!files.length) return;
      event.preventDefault();
      if (canPasteImage(input)) pasteImageFile(input, files.find(file => file.type.startsWith("image/")) || files[0]);
    });
  }
  document.addEventListener("paste", event => {
    if (!document.querySelector("#translate-panel").hidden) return;
    if (event.defaultPrevented || event.target?.closest?.("input,textarea,[contenteditable]")) return;
    const focused=document.activeElement;
    const input=Object.values(imageInputs).find(input => input.area.contains(event.target))
      || Object.values(imageInputs).find(input => input.area.contains(focused))
      || (characterUpload.open && characterUpload.contains(focused) ? imageInputs.character : imageInputs[imagePasteTarget]);
    if (!canPasteImage(input)) return;
    const items = Array.from(event.clipboardData?.items || []);
    const file = items.find(item => item.kind === "file" && item.type.startsWith("image/"))?.getAsFile()
      || Array.from(event.clipboardData?.files || []).find(file => file.type.startsWith("image/"));
    if (file) { event.preventDefault(); pasteImageFile(input, file); }
  });
  document.querySelector("#apply-appearance").addEventListener("click", async () => {
    if (appearanceBusy || (!selectedFile && !descriptionInput.value.trim())) return;
    appearanceBusy = true; updateAppearanceButtons(); appearanceStatus.className=""; appearanceStatus.textContent=appearanceMode.value === "text" ? "Сохраняю описание…" : "Разбираю внешность…";
    try {
      const headers = appearanceMode.value === "image" ? {"X-Appearance-Options":encodeURIComponent(JSON.stringify({species:document.querySelector("#appearance-species").value.trim(),notes:document.querySelector("#appearance-notes").value.trim(),update_existing:document.querySelector("#appearance-same-character").checked}))} : {};
      const profile = await api(appearanceMode.value === "text" ? "PUT" : "POST", selectedFile || {appearance:descriptionInput.value}, "/api/profile/appearance", headers);
      renderAppearance(profile); clearSelection(); descriptionInput.value=""; appearanceStatus.className="ok"; appearanceStatus.textContent="Облик сохранён. Теперь можно вернуться в чат.";
      document.querySelector("#appearance-species").value="";
      document.querySelector("#appearance-notes").value="";
      tg?.HapticFeedback?.notificationOccurred("success");
    } catch(error) { appearanceFail(error); }
    finally { appearanceBusy=false; updateAppearanceButtons(); }
  });
  document.querySelector("#reset-appearance").addEventListener("click", async () => {
    if (appearanceBusy) return;
    appearanceBusy=true; updateAppearanceButtons(); appearanceStatus.className=""; appearanceStatus.textContent="Возвращаю базовый облик…";
    try {
      const profile = await api("DELETE", null, "/api/profile/appearance");
      renderAppearance(profile); clearSelection(); descriptionInput.value=""; appearanceStatus.className="ok"; appearanceStatus.textContent="Базовый облик возвращён.";
      document.querySelector("#appearance-species").value=""; document.querySelector("#appearance-same-character").checked=false;
      document.querySelector("#appearance-notes").value="";
      tg?.HapticFeedback?.notificationOccurred("success");
    } catch(error) { appearanceFail(error); }
    finally { appearanceBusy=false; updateAppearanceButtons(); }
  });
  for (const name of ["rp", "tools", "translate", "markets"]) document.querySelector(`#tab-${name}`).addEventListener("click", () => {
    for (const tab of ["rp", "tools", "translate", "markets"]) { document.querySelector(`#tab-${tab}`).setAttribute("aria-selected", String(tab === name)); document.querySelector(`#${tab}-panel`).hidden = tab !== name; }
    document.body.classList.toggle("translation-view", name === "translate");
    document.body.classList.toggle("markets-view", name === "markets");
  });
  const translationLanguages=[['ru','Русский'],['en','Английский'],['uk','Украинский'],['be','Белорусский'],['de','Немецкий'],['fr','Французский'],['es','Испанский'],['it','Итальянский'],['pt','Португальский'],['pl','Польский'],['nl','Нидерландский'],['cs','Чешский'],['tr','Турецкий'],['kk','Казахский'],['zh','Китайский'],['ja','Японский'],['ko','Корейский'],['ar','Арабский'],['hi','Хинди'],['ka','Грузинский']];
  const translationPanel=document.querySelector('#translate-panel'), translationInput=document.querySelector('#translation-input'), translationOutput=document.querySelector('#translation-output'), translationSource=document.querySelector('#translation-source'), translationTarget=document.querySelector('#translation-target'), translationStatus=document.querySelector('#translation-status'), translationUpload=document.querySelector('#translation-upload'), translationFileInput=document.querySelector('#translation-file'), translationPreview=document.querySelector('#translation-preview');
  for (const [select, items] of [[translationSource,[['auto','Определить язык'],...translationLanguages]],[translationTarget,translationLanguages]]) for(const [code,label] of items) {const option=document.createElement('option'); option.value=code; option.textContent=label; select.append(option);}
  translationSource.value='auto'; translationTarget.value='ru';
  let translationMode='text', translationBusy=false, translationExportBusy=false, translationRevision=0, translationDetected=null, translationPreviewUrl=null, translationDraft='';
  const translationFiles={image:null,document:null};
  const translationLanguageName=code=>translationLanguages.find(item=>item[0]===code)?.[1] || '';
  const updateTranslationButtons=()=>{
    const validText=translationInput.value.trim() && translationInput.value.length<=5000;
    document.querySelector('#translation-submit').disabled=translationBusy || (translationMode==='text' ? !validText : !translationFiles[translationMode]);
    document.querySelector('#translation-swap').disabled=translationBusy || translationMode!=='text' || (translationSource.value==='auto' && !translationDetected) || translationOutput.value.length>5000;
    for(const id of ['translation-copy','translation-txt','translation-docx']) document.querySelector(`#${id}`).disabled=translationBusy || translationExportBusy || !translationOutput.value;
    for(const id of ['translation-choose','translation-paste']) document.querySelector(`#${id}`).disabled=translationBusy;
    document.querySelector('#translation-count').textContent=translationMode==='text' ? `${translationInput.value.length} / 5000` : (translationInput.value ? `${translationInput.value.length} символов` : '');
  };
  const invalidateTranslation=()=>{
    translationRevision++; translationDetected=null; translationOutput.value=''; document.querySelector('#translation-detected').textContent='';
    translationStatus.className='tool-status'; translationStatus.textContent=translationBusy ? 'Запрос ещё выполняется. Изменённый текст переведи повторно.' : '';
    updateTranslationButtons();
  };
  const renderTranslationFile=()=>{
    if(translationPreviewUrl) {URL.revokeObjectURL(translationPreviewUrl); translationPreviewUrl=null;}
    const file=translationFiles[translationMode]; translationPreview.hidden=true; translationPreview.removeAttribute('src');
    document.querySelector('#translation-file-name').textContent=file ? `${file.name} · ${(file.size/1024/1024).toFixed(2)} МБ` : '';
    if(file && translationMode==='image') {translationPreviewUrl=URL.createObjectURL(file); translationPreview.src=translationPreviewUrl; translationPreview.hidden=false;}
  };
  const chooseTranslationFile=file=>{
    if(!file || translationMode==='text' || translationBusy) return;
    if(!file.size || file.size>20*1024*1024) {translationStatus.className='tool-status error'; translationStatus.textContent='Выбери файл размером до 20 МБ.'; return;}
    if(translationMode==='image' && !['image/jpeg','image/png','image/webp'].includes(file.type) && !/\.(jpe?g|png|webp)$/i.test(file.name)) {translationStatus.className='tool-status error'; translationStatus.textContent='Поддерживаются JPEG, PNG и WebP.'; return;}
    translationFiles[translationMode]=file; translationInput.value=''; invalidateTranslation(); renderTranslationFile();
  };
  for(const button of document.querySelectorAll('[data-translation-mode]')) button.addEventListener('click',()=>{
    if(translationMode==='text') translationDraft=translationInput.value;
    translationMode=button.dataset.translationMode; invalidateTranslation();
    for(const other of document.querySelectorAll('[data-translation-mode]')) other.setAttribute('aria-pressed',String(other===button));
    translationInput.readOnly=translationMode!=='text'; translationInput.value=translationMode==='text' ? translationDraft : '';
    translationInput.placeholder=translationMode==='text' ? 'Введите текст' : 'Здесь появится прочитанный текст';
    document.querySelector('#translation-input-label').textContent=translationMode==='text' ? 'Исходный текст' : 'Текст из файла';
    translationUpload.hidden=translationMode==='text'; document.querySelector('#translation-paste').hidden=translationMode!=='image';
    translationFileInput.accept=translationMode==='image' ? 'image/jpeg,image/png,image/webp' : '.pdf,.docx,.xlsx,.txt,.md,.csv,.json,.html,.xml,.yaml,.yml,.log,.py,.js,.ts';
    document.querySelector('#translation-upload-hint').textContent=translationMode==='image' ? 'Перетащите картинку или вставьте из буфера. JPEG, PNG, WebP до 20 МБ.' : 'Перетащите документ. PDF, DOCX, XLSX и текстовые файлы до 20 МБ; до 60000 символов, 40 страниц PDF или 4 страниц скана. Результат — текст перевода.';
    renderTranslationFile(); updateTranslationButtons();
  });
  translationInput.addEventListener('input',()=>{if(translationMode==='text') translationDraft=translationInput.value; invalidateTranslation();});
  for(const select of [translationSource,translationTarget]) select.addEventListener('change',invalidateTranslation);
  document.querySelector('#translation-clear').addEventListener('click',()=>{translationInput.value=''; if(translationMode==='text') translationDraft=''; else translationFiles[translationMode]=null; invalidateTranslation(); renderTranslationFile();});
  document.querySelector('#translation-swap').addEventListener('click',()=>{
    const source=translationSource.value==='auto' ? translationDetected : translationSource.value;
    if(!source || translationBusy || translationMode!=='text' || translationOutput.value.length>5000) return;
    translationInput.value=translationOutput.value || translationInput.value; translationDraft=translationInput.value;
    translationSource.value=translationTarget.value; translationTarget.value=source; invalidateTranslation();
  });
  document.querySelector('#translation-choose').addEventListener('click',()=>{translationFileInput.value=''; translationFileInput.click();});
  translationFileInput.addEventListener('change',()=>chooseTranslationFile(translationFileInput.files[0]));
  translationUpload.addEventListener('dragover',event=>{if(Array.from(event.dataTransfer?.types || []).includes('Files')) {event.preventDefault(); if(!translationBusy) translationUpload.classList.add('drag-over');}});
  translationUpload.addEventListener('dragleave',()=>translationUpload.classList.remove('drag-over'));
  translationUpload.addEventListener('drop',event=>{translationUpload.classList.remove('drag-over'); const file=event.dataTransfer?.files?.[0]; if(file) {event.preventDefault(); chooseTranslationFile(file);}});
  translationPanel.addEventListener('paste',event=>{if(translationMode!=='image' || translationBusy) return; const file=Array.from(event.clipboardData?.files || []).find(item=>item.type.startsWith('image/')) || Array.from(event.clipboardData?.items || []).find(item=>item.kind==='file' && item.type.startsWith('image/'))?.getAsFile(); if(file) {event.preventDefault(); chooseTranslationFile(file);}});
  document.querySelector('#translation-paste').addEventListener('click',async()=>{
    const revision=translationRevision;
    try {
      if(!navigator.clipboard?.read) throw new Error('Используй Ctrl+V в области загрузки или выбери файл.');
      const items=await navigator.clipboard.read(); const item=items.find(item=>item.types.some(type=>['image/png','image/jpeg','image/webp'].includes(type)));
      if(!item) throw new Error('В буфере нет картинки.');
      const type=item.types.find(type=>['image/png','image/jpeg','image/webp'].includes(type)); const blob=await item.getType(type);
      if(translationMode==='image' && translationRevision===revision) chooseTranslationFile(new File([blob],`clipboard.${type.split('/')[1]}`,{type}));
    } catch(error) {if(translationMode==='image' && translationRevision===revision) {translationStatus.className='tool-status error'; translationStatus.textContent=error.message || 'Не удалось прочитать буфер. Выбери файл.';}}
  });
  document.querySelector('#translation-submit').addEventListener('click',async()=>{
    if(translationBusy) return;
    const revision=translationRevision, mode=translationMode, source=translationSource.value, target=translationTarget.value, text=translationInput.value, file=translationFiles[mode];
    translationBusy=true; updateTranslationButtons(); translationStatus.className='tool-status'; translationStatus.textContent=mode==='text' ? 'Перевожу…' : 'Читаю файл и перевожу… Большой документ может занять несколько минут.';
    try {
      if(!initData) throw new Error('Открой переводчик внутри Telegram.');
      const result=mode==='text' ? await api('POST',{text,source,target},'/api/tools/translate') : await api('POST',file,'/api/tools/translate/file',{'X-Translation-Options':encodeURIComponent(JSON.stringify({mode,source,target,file_name:file.name.slice(0,160)}))});
      if(revision!==translationRevision) return;
      translationOutput.value=result.text; translationDetected=result.detected_source;
      if(mode!=='text') translationInput.value=result.source_text || '';
      document.querySelector('#translation-detected').textContent=source==='auto' && translationDetected ? `Определён язык: ${translationLanguageName(translationDetected)}` : '';
      translationStatus.textContent='Готово.'; translationStatus.className='tool-status ok';
    } catch(error) {if(revision===translationRevision) {translationStatus.className='tool-status error'; translationStatus.textContent=error.message;}}
    finally {translationBusy=false; if(revision!==translationRevision) translationStatus.textContent='Данные изменились. Нажми «Перевести» для нового запроса.'; updateTranslationButtons();}
  });
  document.querySelector('#translation-copy').addEventListener('click',async()=>{try {await navigator.clipboard.writeText(translationOutput.value); translationStatus.className='tool-status ok'; translationStatus.textContent='Перевод скопирован.';} catch {translationOutput.focus(); translationOutput.select(); translationStatus.textContent='Перевод выделен. Скопируй его вручную.';}});
  const downloadTranslation=(blob,extension)=>{const url=URL.createObjectURL(blob), link=document.createElement('a'); link.href=url; link.download=`translation-${translationTarget.value}.${extension}`; document.body.append(link); link.click(); link.remove(); setTimeout(()=>URL.revokeObjectURL(url),60000);};
  document.querySelector('#translation-txt').addEventListener('click',()=>{if(translationOutput.value) downloadTranslation(new Blob(['\ufeff',translationOutput.value],{type:'text/plain;charset=utf-8'}),'txt');});
  document.querySelector('#translation-docx').addEventListener('click',async()=>{
    if(!translationOutput.value || translationExportBusy) return;
    const text=translationOutput.value, revision=translationRevision; translationExportBusy=true; updateTranslationButtons(); translationStatus.textContent='Создаю DOCX…';
    try {
      const response=await fetch('/api/tools/translate/export',{method:'POST',headers:{'X-Telegram-Init-Data':initData,'Content-Type':'application/json'},body:JSON.stringify({text})});
      if(!response.ok) throw new Error(await response.text());
      const blob=await response.blob(); if(revision===translationRevision) {downloadTranslation(blob,'docx'); translationStatus.className='tool-status ok'; translationStatus.textContent='DOCX готов к скачиванию.';}
    } catch(error) {if(revision===translationRevision) {translationStatus.className='tool-status error'; translationStatus.textContent=error.message;}}
    finally {translationExportBusy=false; updateTranslationButtons();}
  });
  updateTranslationButtons();
  const toolCall = async (button, statusId, tool, body, done) => {
    const output=document.querySelector(statusId); button.disabled=true; output.className="tool-status"; output.textContent="Обрабатываю…";
    try { if (!initData) throw new Error("Открой панель внутри Telegram."); const result=await api("POST", body, `/api/tools/${tool}`); done(result, output); }
    catch(error) { output.className="tool-status error"; output.textContent=error.message; }
    finally { button.disabled=false; }
  };
  let copiedId=null;
  const showId = (result, output) => { copiedId=result.id ?? null; document.querySelector("#copy-id").hidden=copiedId===null; output.textContent=result.message || `${result.title}\nID: ${result.id}`; };
  for (const [selector, target] of [["#my-id","self"],["#bot-id","bot"],["#select-id","select"],["#lookup-id",null]]) document.querySelector(selector).addEventListener("click", e => toolCall(e.currentTarget,"#id-status","id",{target:target || document.querySelector("#id-target").value.trim()},showId));
  document.querySelector("#copy-id").addEventListener("click", async () => { try { await navigator.clipboard.writeText(String(copiedId)); document.querySelector("#id-status").textContent=`ID ${copiedId} скопирован.`; } catch { document.querySelector("#id-status").textContent=`ID: ${copiedId}. Выдели текст и скопируй.`; } });
  document.querySelector("#download-video").addEventListener("click", e => toolCall(e.currentTarget,"#download-status","download",{url:document.querySelector("#download-url").value},(result,output)=>{output.textContent=result.message;}));
  const conversionFormats = {video:['mp4','avi','mov','mkv'],audio:['mp3','wav','flac']};
  const conversionFormat = document.querySelector('#conversion-format'), conversionFileInput = document.querySelector('#conversion-file');
  const conversionStatus = document.querySelector('#conversion-status'), conversionUpload = document.querySelector('#conversion-upload');
  let conversionMode='video', conversionFile=null, conversionResult=null, conversionBusy=false;
  const conversionPlayers=[document.querySelector('#conversion-video'),document.querySelector('#conversion-audio')];
  const conversionResetResult=()=>{
    if(conversionResult) api('DELETE',null,`/api/tools/convert/${conversionResult.token}`).catch(()=>{});
    conversionResult=null;
    document.querySelector('#conversion-result').hidden=true;
    for(const player of conversionPlayers) {player.pause();player.removeAttribute('src');player.load();player.hidden=true;}
  };
  const conversionUpdate=()=>{
    for(const selector of ['#conversion-choose','#conversion-format','#conversion-resolution','#conversion-bitrate']) document.querySelector(selector).disabled=conversionBusy;
    for(const button of document.querySelectorAll('[data-conversion-mode]')) button.disabled=conversionBusy;
    document.querySelector('#conversion-submit').disabled=conversionBusy || !conversionFile;
    document.querySelector('#conversion-clear').disabled=conversionBusy || !(conversionFile || conversionResult);
    document.querySelector('#conversion-resolution-field').hidden=conversionMode!=='video';
    document.querySelector('#conversion-bitrate-field').hidden=conversionMode!=='audio' || conversionFormat.value!=='mp3';
    conversionUpload.setAttribute('aria-busy',String(conversionBusy));
    document.querySelector('#conversion-hint').textContent=conversionMode==='video' ? 'Файл до 100 МБ и 10 минут. Видео сохраняется со звуком, если он есть; до 30 кадров/с. Обработка — до четырёх минут.' : 'Аудио или видео до 100 МБ и 10 минут. Из видео извлекается звук. WAV и FLAC сохраняются без сжатия с потерями; исходное качество не восстанавливается. Результат — до 150 МБ.';
  };
  const setConversionMode=mode=>{
    if(conversionBusy) return;
    conversionMode=mode; conversionResetResult();
    conversionFormat.replaceChildren(...conversionFormats[mode].map(format=>new Option(format.toUpperCase(),format)));
    for(const button of document.querySelectorAll('[data-conversion-mode]')) button.setAttribute('aria-pressed',String(button.dataset.conversionMode===mode));
    conversionStatus.textContent='';conversionUpdate();
  };
  const chooseConversionFile=file=>{
    if(!file || conversionBusy) return;
    if(file.size===0 || file.size>100*1024*1024) {conversionStatus.className='tool-status error';conversionStatus.textContent='Нужен непустой файл до 100 МБ.';return;}
    conversionResetResult();conversionFile=file;
    document.querySelector('#conversion-file-name').textContent=`${file.name} · ${(file.size/1024/1024).toFixed(2)} МБ`;
    conversionStatus.textContent='';conversionUpdate();
  };
  for(const button of document.querySelectorAll('[data-conversion-mode]')) button.addEventListener('click',()=>setConversionMode(button.dataset.conversionMode));
  for(const select of [conversionFormat,document.querySelector('#conversion-resolution'),document.querySelector('#conversion-bitrate')]) select.addEventListener('change',()=>{conversionResetResult();conversionStatus.textContent='';conversionUpdate();});
  document.querySelector('#conversion-choose').addEventListener('click',()=>{conversionFileInput.value='';conversionFileInput.click();});
  conversionFileInput.addEventListener('change',()=>chooseConversionFile(conversionFileInput.files[0]));
  conversionUpload.addEventListener('dragover',event=>{if(Array.from(event.dataTransfer?.types || []).includes('Files')) {event.preventDefault();if(!conversionBusy) conversionUpload.classList.add('drag-over');}});
  conversionUpload.addEventListener('dragleave',()=>conversionUpload.classList.remove('drag-over'));
  conversionUpload.addEventListener('drop',event=>{conversionUpload.classList.remove('drag-over');if(event.dataTransfer?.files?.length) {event.preventDefault();chooseConversionFile(event.dataTransfer.files[0]);}});
  document.querySelector('#conversion-clear').addEventListener('click',()=>{conversionResetResult();conversionFile=null;conversionFileInput.value='';document.querySelector('#conversion-file-name').textContent='';conversionStatus.textContent='';conversionUpdate();});
  document.querySelector('#conversion-submit').addEventListener('click',async()=>{
    if(conversionBusy || !conversionFile) return;
    conversionResetResult();conversionBusy=true;conversionUpdate();conversionStatus.className='tool-status';conversionStatus.textContent='Загружаю файл…';
    const progress=document.querySelector('#conversion-progress');progress.hidden=false;progress.value=0;
    try {
      const result=await new Promise((resolve,reject)=>{
        const xhr=new XMLHttpRequest();xhr.open('POST','/api/tools/convert/file');xhr.timeout=250000;xhr.responseType='json';
        xhr.setRequestHeader('X-Telegram-Init-Data',initData);xhr.setRequestHeader('Content-Type',conversionFile.type || 'application/octet-stream');
        xhr.setRequestHeader('X-Conversion-Options',encodeURIComponent(JSON.stringify({mode:conversionMode,format:conversionFormat.value,resolution:document.querySelector('#conversion-resolution').value,bitrate:Number(document.querySelector('#conversion-bitrate').value),file_name:conversionFile.name})));
        xhr.upload.onprogress=event=>{if(event.lengthComputable) progress.value=Math.round(event.loaded/event.total*100);};
        xhr.upload.onload=()=>{progress.removeAttribute('value');conversionStatus.textContent='Преобразую файл… Это может занять до четырёх минут.';};
        xhr.onload=()=>xhr.status>=200 && xhr.status<300 && xhr.response ? resolve(xhr.response) : reject(new Error(xhr.status===401 ? 'Срок доступа истёк. Закрой Mini App и открой заново.' : xhr.status===429 ? 'Конвертер занят или запросы слишком частые. Попробуй позже.' : 'Не удалось преобразовать файл. Проверь формат, размер и длительность.'));
        xhr.onerror=()=>reject(new Error('Соединение прервалось. Попробуй ещё раз.'));xhr.ontimeout=()=>reject(new Error('Обработка заняла слишком много времени. Выбери меньший файл.'));
        xhr.send(conversionFile);
      });
      conversionResult=result;
      document.querySelector('#conversion-result-name').textContent=result.name;
      document.querySelector('#conversion-result-info').textContent=`${result.format.toUpperCase()} · ${(result.bytes/1024/1024).toFixed(2)} МБ · ${Math.floor(result.duration/60)}:${String(Math.round(result.duration%60)).padStart(2,'0')}`;
      const player=conversionPlayers[conversionMode==='video' ? 0 : 1];player.src=result.preview_url;player.hidden=false;
      document.querySelector('#conversion-preview-hint').textContent=result.format==='mp4' || result.format==='mp3' || result.format==='wav' ? 'Можно посмотреть или послушать результат перед сохранением.' : 'Для плеера подготовлено отдельное превью. Сохранится файл в выбранном формате.';
      document.querySelector('#conversion-result').hidden=false;conversionStatus.className='tool-status ok';conversionStatus.textContent='Готово. Посмотри результат и сохрани файл.';
    } catch(error) {conversionStatus.className='tool-status error';conversionStatus.textContent=error.message;}
    finally {conversionBusy=false;progress.hidden=true;conversionUpdate();}
  });
  for(const player of conversionPlayers) player.addEventListener('error',()=>{if(conversionResult) document.querySelector('#conversion-preview-hint').textContent='Превью не удалось открыть или оно устарело. Попробуй сохранить файл либо преобразовать его заново.';});
  document.querySelector('#conversion-save').addEventListener('click',()=>{
    if(!conversionResult) return;
    const result=conversionResult;
    const download=()=>{const link=document.createElement('a');link.href=result.download_url;link.download=result.name;link.rel='noreferrer';document.body.append(link);link.click();link.remove();};
    if(tg?.isVersionAtLeast?.('8.0') && tg.downloadFile) {try {tg.downloadFile({url:new URL(result.download_url,location.href).href,file_name:result.name},accepted=>{if(!accepted) {conversionStatus.textContent='Сохранение отменено. Можно нажать «Сохранить файл» ещё раз.';}});} catch {download();}}
    else download();
  });
  setConversionMode('video');
  const fail = e => { status.className="error"; status.textContent = initData ? `Ошибка: ${e.message}` : "Открой панель кнопкой внутри Telegram."; };
  updateAppearanceButtons();
  api().then(render).catch(error => {
    fail(error);
    document.querySelector("#appearance").textContent = "Не удалось загрузить облик.";
    appearanceFail(error);
  });
  document.querySelector("#save").addEventListener("click", async e => { const button=e.currentTarget; settingsBusy=true; updateAppearanceButtons(); status.textContent="Сохраняю…"; status.className=""; try { const p=await api("PATCH",{content_mode:document.querySelector("#content-mode").value,roleplay_configuration:document.querySelector("#configuration").value,roleplay_allow_profanity:document.querySelector("#rp-allow-profanity").checked,roleplay_character:document.querySelector("#character").value,expected_character:characterSnapshot,roleplay_preferences:document.querySelector("#preferences").value,roleplay_boundaries:document.querySelector("#boundaries").value,roleplay_stopword:document.querySelector("#stopword").value}); render(p); status.className="ok"; status.textContent="Сохранено."; tg?.HapticFeedback?.notificationOccurred("success"); } catch(err) { fail(err); } finally { settingsBusy=false; updateAppearanceButtons(); } });
  document.querySelector("#clear-rp").addEventListener("click", async e => { if (!window.confirm("Удалить персонажа, предпочтения, границы и распознанные RP-темы?")) return; const button=e.currentTarget; button.disabled=true; try { const p=await api("DELETE",null,"/api/profile/roleplay"); render(p); status.className="ok"; status.textContent="RP-профиль очищен. Остальная память сохранена."; } catch(err) { fail(err); } finally { button.disabled=false; } });
