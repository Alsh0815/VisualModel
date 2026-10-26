/**
 * VisualModel Dataset Studio - Frontend Application
 */

// アプリケーション状態
const state = {
  currentSubset: '',
  subsets: [],
  items: [],
  filteredItems: [],
  selectedIndex: -1,
  selectedItem: null,
  isDirty: false,
  originalPrompt: '',
  activeWordFilter: null,
  wordStatsData: null,
  preparePollingInterval: null
};

// DOM要素の参照キャッシュ
const dom = {
  subsetSelect: document.getElementById('subsetSelect'),
  refreshBtn: document.getElementById('refreshBtn'),
  headerStats: document.getElementById('headerStats'),
  statImageCount: document.getElementById('statImageCount'),
  statAvgChars: document.getElementById('statAvgChars'),
  statTotalChars: document.getElementById('statTotalChars'),
  
  // ギャラリー
  galleryGrid: document.getElementById('galleryGrid'),
  searchInput: document.getElementById('searchInput'),
  clearSearchBtn: document.getElementById('clearSearchBtn'),
  sortSelect: document.getElementById('sortSelect'),
  gridSizeSlider: document.getElementById('gridSizeSlider'),
  filteredCountText: document.getElementById('filteredCountText'),
  activeWordFilter: document.getElementById('activeWordFilter'),
  activeFilterWord: document.getElementById('activeFilterWord'),
  clearWordFilterBtn: document.getElementById('clearWordFilterBtn'),

  // エディタ
  currentFilename: document.getElementById('currentFilename'),
  currentExt: document.getElementById('currentExt'),
  saveStatusBadge: document.getElementById('saveStatusBadge'),
  currentPositionIndex: document.getElementById('currentPositionIndex'),
  prevBtn: document.getElementById('prevBtn'),
  nextBtn: document.getElementById('nextBtn'),
  previewImage: document.getElementById('previewImage'),
  noImageSelected: document.getElementById('noImageSelected'),
  zoomImageBtn: document.getElementById('zoomImageBtn'),
  imageMetaBar: document.getElementById('imageMetaBar'),
  metaResolution: document.getElementById('metaResolution'),
  metaAspect: document.getElementById('metaAspect'),
  metaFileSize: document.getElementById('metaFileSize'),
  metaTensorBadge: document.getElementById('metaTensorBadge'),

  // リアルタイムカウンター
  countTotal: document.getElementById('countTotal'),
  countNoSpaces: document.getElementById('countNoSpaces'),
  countWords: document.getElementById('countWords'),
  countLines: document.getElementById('countLines'),
  countTokens: document.getElementById('countTokens'),
  tokenLimitBadge: document.getElementById('tokenLimitBadge'),
  promptTextarea: document.getElementById('promptTextarea'),

  // エディタアクション
  savePromptBtn: document.getElementById('savePromptBtn'),
  resetPromptBtn: document.getElementById('resetPromptBtn'),
  copyPromptBtn: document.getElementById('copyPromptBtn'),

  // モーダル: 単語統計
  openWordStatsBtn: document.getElementById('openWordStatsBtn'),
  wordStatsModal: document.getElementById('wordStatsModal'),
  closeWordStatsBtn: document.getElementById('closeWordStatsBtn'),
  closeWordStatsFooterBtn: document.getElementById('closeWordStatsFooterBtn'),
  statsTotalTokens: document.getElementById('statsTotalTokens'),
  statsUniqueWords: document.getElementById('statsUniqueWords'),
  statsTotalDocs: document.getElementById('statsTotalDocs'),
  statsEngineName: document.getElementById('statsEngineName'),
  wordStatsTableBody: document.getElementById('wordStatsTableBody'),
  wordCloudContainer: document.getElementById('wordCloudContainer'),
  wordSearchInput: document.getElementById('wordSearchInput'),
  filterStopwordsCheck: document.getElementById('filterStopwordsCheck'),
  minWordLenSelect: document.getElementById('minWordLenSelect'),
  exportCsvBtn: document.getElementById('exportCsvBtn'),

  // モーダル: 一括置換
  openBatchReplaceBtn: document.getElementById('openBatchReplaceBtn'),
  batchReplaceModal: document.getElementById('batchReplaceModal'),
  closeBatchReplaceBtn: document.getElementById('closeBatchReplaceBtn'),
  replaceTargetSubset: document.getElementById('replaceTargetSubset'),
  findWordInput: document.getElementById('findWordInput'),
  replaceWordInput: document.getElementById('replaceWordInput'),
  replacePreviewArea: document.getElementById('replacePreviewArea'),
  replaceCount: document.getElementById('replaceCount'),
  replacePreviewList: document.getElementById('replacePreviewList'),
  previewReplaceBtn: document.getElementById('previewReplaceBtn'),
  executeReplaceBtn: document.getElementById('executeReplaceBtn'),

  // モーダル: テンソル生成
  runPrepareBtn: document.getElementById('runPrepareBtn'),
  prepareModal: document.getElementById('prepareModal'),
  closePrepareModalBtn: document.getElementById('closePrepareModalBtn'),
  closePrepareFooterBtn: document.getElementById('closePrepareFooterBtn'),
  startPrepareBtn: document.getElementById('startPrepareBtn'),
  prepareStatusText: document.getElementById('prepareStatusText'),
  prepareSpinner: document.getElementById('prepareSpinner'),
  prepareLogs: document.getElementById('prepareLogs'),

  // フルスクリーン画像
  fullscreenModal: document.getElementById('fullscreenModal'),
  fullscreenImage: document.getElementById('fullscreenImage'),
  closeFullscreenBtn: document.getElementById('closeFullscreenBtn'),

  // トースト
  toastContainer: document.getElementById('toastContainer'),
  paneResizer: document.getElementById('paneResizer')
};

// ==========================================================================
// 初期化
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  loadSubsets();
});

function initEventListeners() {
  // サブセット切替
  dom.subsetSelect.addEventListener('change', (e) => {
    switchSubset(e.target.value);
  });

  dom.refreshBtn.addEventListener('click', () => {
    loadSubsets(state.currentSubset);
  });

  // ギャラリー検索 & フィルタ
  dom.searchInput.addEventListener('input', () => {
    dom.clearSearchBtn.style.display = dom.searchInput.value ? 'block' : 'none';
    applyFiltersAndSort();
  });

  dom.clearSearchBtn.addEventListener('click', () => {
    dom.searchInput.value = '';
    dom.clearSearchBtn.style.display = 'none';
    applyFiltersAndSort();
  });

  dom.clearWordFilterBtn.addEventListener('click', () => {
    clearWordFilter();
  });

  dom.sortSelect.addEventListener('change', () => {
    applyFiltersAndSort();
  });

  dom.gridSizeSlider.addEventListener('input', (e) => {
    document.documentElement.style.setProperty('--item-min-width', `${e.target.value}px`);
  });

  // プロンプト編集・リアルタイムカウント
  dom.promptTextarea.addEventListener('input', () => {
    updateRealtimeCounts();
    checkDirtyState();
  });

  // 保存 & リセット
  dom.savePromptBtn.addEventListener('click', saveCurrentPrompt);
  dom.resetPromptBtn.addEventListener('click', resetCurrentPrompt);
  dom.copyPromptBtn.addEventListener('click', copyCurrentPrompt);

  // ナビゲーション
  dom.prevBtn.addEventListener('click', () => navigateImage(-1));
  dom.nextBtn.addEventListener('click', () => navigateImage(1));

  // キーボードショートカット
  window.addEventListener('keydown', handleGlobalKeydown);

  // 画像ズーム
  dom.zoomImageBtn.addEventListener('click', openFullscreenImage);
  dom.closeFullscreenBtn.addEventListener('click', closeFullscreenImage);
  dom.fullscreenModal.addEventListener('click', (e) => {
    if (e.target === dom.fullscreenModal) closeFullscreenImage();
  });

  // 単語統計モーダル
  dom.openWordStatsBtn.addEventListener('click', openWordStatsModal);
  dom.closeWordStatsBtn.addEventListener('click', closeWordStatsModal);
  dom.closeWordStatsFooterBtn.addEventListener('click', closeWordStatsModal);
  dom.wordStatsModal.addEventListener('click', (e) => {
    if (e.target === dom.wordStatsModal) closeWordStatsModal();
  });
  dom.wordSearchInput.addEventListener('input', renderWordStats);
  dom.filterStopwordsCheck.addEventListener('change', fetchWordStats);
  dom.minWordLenSelect.addEventListener('change', fetchWordStats);
  dom.exportCsvBtn.addEventListener('click', exportWordStatsCsv);

  // 単語統計タブ切り替え
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
      e.target.classList.add('active');
      const targetId = e.target.getAttribute('data-tab');
      document.getElementById(targetId).style.display = 'block';
    });
  });

  // 一括置換モーダル
  dom.openBatchReplaceBtn.addEventListener('click', openBatchReplaceModal);
  dom.closeBatchReplaceBtn.addEventListener('click', closeBatchReplaceModal);
  dom.batchReplaceModal.addEventListener('click', (e) => {
    if (e.target === dom.batchReplaceModal) closeBatchReplaceModal();
  });
  dom.previewReplaceBtn.addEventListener('click', () => executeBatchReplace(true));
  dom.executeReplaceBtn.addEventListener('click', () => executeBatchReplace(false));

  // テンソル生成モーダル
  dom.runPrepareBtn.addEventListener('click', openPrepareModal);
  dom.closePrepareModalBtn.addEventListener('click', closePrepareModal);
  dom.closePrepareFooterBtn.addEventListener('click', closePrepareModal);
  dom.prepareModal.addEventListener('click', (e) => {
    if (e.target === dom.prepareModal) closePrepareModal();
  });
  dom.startPrepareBtn.addEventListener('click', startPrepareText);

  // パネルリサイズ機能
  initPaneResizer();
}

// ==========================================================================
// データ通信 & サブセット管理
// ==========================================================================
async function loadSubsets(preferredSubset = '') {
  try {
    const res = await fetch('/api/subsets');
    const data = await res.json();
    state.subsets = data.subsets || [];

    dom.subsetSelect.innerHTML = '';
    if (state.subsets.length === 0) {
      dom.subsetSelect.innerHTML = '<option value="" disabled selected>サブセットがありません</option>';
      renderEmptyGallery('datasets フォルダ内にサブフォルダが見つかりません');
      return;
    }

    state.subsets.forEach(sub => {
      const opt = document.createElement('option');
      opt.value = sub.name;
      opt.textContent = `${sub.name} (${sub.image_count}枚)`;
      dom.subsetSelect.appendChild(opt);
    });

    const target = preferredSubset && state.subsets.some(s => s.name === preferredSubset)
      ? preferredSubset
      : state.subsets[0].name;

    dom.subsetSelect.value = target;
    await switchSubset(target);
  } catch (err) {
    showToast(`データセット一覧の取得に失敗しました: ${err.message}`, 'error');
  }
}

async function switchSubset(subsetName) {
  if (state.isDirty) {
    if (!confirm('未保存のプロンプト変更があります。破棄して移動しますか？')) {
      dom.subsetSelect.value = state.currentSubset;
      return;
    }
  }

  state.currentSubset = subsetName;
  state.selectedIndex = -1;
  state.selectedItem = null;
  state.isDirty = false;
  state.activeWordFilter = null;
  dom.activeWordFilter.style.display = 'none';

  renderEmptyGallery('読み込み中...');

  try {
    const res = await fetch(`/api/dataset?subset=${encodeURIComponent(subsetName)}`);
    const data = await res.json();
    state.items = data.items || [];

    updateHeaderStats();
    applyFiltersAndSort();

    if (state.filteredItems.length > 0) {
      selectItem(0);
    } else {
      clearEditor();
    }
  } catch (err) {
    showToast(`データセットの読み込みに失敗しました: ${err.message}`, 'error');
    renderEmptyGallery('データの読み込みに失敗しました');
  }
}

function updateHeaderStats() {
  const total = state.items.length;
  let totalChars = 0;
  state.items.forEach(item => {
    totalChars += item.char_count || 0;
  });
  const avgChars = total > 0 ? Math.round(totalChars / total) : 0;

  dom.statImageCount.textContent = total.toLocaleString();
  dom.statAvgChars.textContent = avgChars.toLocaleString();
  dom.statTotalChars.textContent = totalChars.toLocaleString();
}

// ==========================================================================
// ギャラリーのフィルタ・ソート・レンダリング
// ==========================================================================
function applyFiltersAndSort() {
  const query = dom.searchInput.value.trim().toLowerCase();
  const sort = dom.sortSelect.value;
  const wordFilter = state.activeWordFilter ? state.activeWordFilter.toLowerCase() : null;

  state.filteredItems = state.items.filter(item => {
    // 単語フィルタ
    if (wordFilter) {
      if (!item.prompt.toLowerCase().includes(wordFilter)) {
        return false;
      }
    }
    // 検索語句
    if (query) {
      const matchName = item.filename.toLowerCase().includes(query);
      const matchPrompt = item.prompt.toLowerCase().includes(query);
      if (!matchName && !matchPrompt) return false;
    }
    return true;
  });

  // ソート
  state.filteredItems.sort((a, b) => {
    switch (sort) {
      case 'name_asc': return a.filename.localeCompare(b.filename, undefined, { numeric: true });
      case 'name_desc': return b.filename.localeCompare(a.filename, undefined, { numeric: true });
      case 'chars_desc': return (b.char_count || 0) - (a.char_count || 0);
      case 'chars_asc': return (a.char_count || 0) - (b.char_count || 0);
      case 'res_desc': return (b.width * b.height) - (a.width * a.height);
      case 'mtime_desc': return (b.mtime || 0) - (a.mtime || 0);
      default: return 0;
    }
  });

  dom.filteredCountText.textContent = `表示中: ${state.filteredItems.length} / ${state.items.length} 件`;

  renderGalleryGrid();

  // 現在の選択アイテムがフィルタ結果内にあれば選択維持、なければ先頭を選択
  if (state.selectedItem) {
    const newIdx = state.filteredItems.findIndex(i => i.id === state.selectedItem.id);
    if (newIdx !== -1) {
      state.selectedIndex = newIdx;
      highlightActiveCard();
      updateNavCounter();
    } else if (state.filteredItems.length > 0) {
      selectItem(0);
    } else {
      clearEditor();
    }
  } else if (state.filteredItems.length > 0) {
    selectItem(0);
  } else {
    clearEditor();
  }
}

function renderGalleryGrid() {
  dom.galleryGrid.innerHTML = '';
  if (state.filteredItems.length === 0) {
    renderEmptyGallery('該当する画像がありません');
    return;
  }

  const fragment = document.createDocumentFragment();

  state.filteredItems.forEach((item, index) => {
    const card = document.createElement('div');
    card.className = `gallery-card ${index === state.selectedIndex ? 'active' : ''}`;
    card.id = `card-${item.id}`;

    const imgUrl = `/api/image?subset=${encodeURIComponent(state.currentSubset)}&filename=${encodeURIComponent(item.filename)}`;
    const resText = item.width ? `${item.width}×${item.height}` : '-';

    card.innerHTML = `
      <div class="card-thumb-wrapper">
        <img src="${imgUrl}" alt="${escapeHtml(item.filename)}" loading="lazy">
        <div class="card-badge-container">
          <span class="card-char-badge">${item.char_count}字</span>
          <span class="card-tensor-dot ${item.has_tensor ? '' : 'missing'}" title="${item.has_tensor ? 'テンソル生成済' : 'テンソル未生成'}"></span>
        </div>
      </div>
      <div class="card-info">
        <div class="card-filename" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</div>
        <div class="card-meta">
          <span>${resText}</span>
          <span>${formatFileSize(item.file_size)}</span>
        </div>
      </div>
    `;

    card.addEventListener('click', () => {
      if (index !== state.selectedIndex) {
        selectItem(index);
      }
    });

    fragment.appendChild(card);
  });

  dom.galleryGrid.appendChild(fragment);
}

function renderEmptyGallery(message) {
  dom.galleryGrid.innerHTML = `
    <div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width: 48px; height: 48px; opacity: 0.3;">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <circle cx="8.5" cy="8.5" r="1.5"></circle>
        <polyline points="21 15 16 10 5 21"></polyline>
      </svg>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function highlightActiveCard() {
  document.querySelectorAll('.gallery-card.active').forEach(c => c.classList.remove('active'));
  if (state.selectedItem) {
    const activeCard = document.getElementById(`card-${state.selectedItem.id}`);
    if (activeCard) {
      activeCard.classList.add('active');
      activeCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
}

// ==========================================================================
// アイテム選択 ＆ エディタ
// ==========================================================================
function selectItem(index) {
  if (index < 0 || index >= state.filteredItems.length) return;

  if (state.isDirty) {
    // 簡易確認
    const proceed = confirm('現在のプロンプト編集内容が未保存です。破棄して移動しますか？\n（[OK]: 破棄して移動, [キャンセル]: 編集を継続）');
    if (!proceed) return;
  }

  state.selectedIndex = index;
  state.selectedItem = state.filteredItems[index];
  state.isDirty = false;
  state.originalPrompt = state.selectedItem.prompt || '';

  // エディタへの値反映
  dom.currentFilename.textContent = state.selectedItem.filename;
  dom.currentExt.textContent = (state.selectedItem.ext || '').replace('.', '').toUpperCase();
  dom.promptTextarea.value = state.originalPrompt;
  dom.promptTextarea.disabled = false;
  dom.savePromptBtn.disabled = true;
  dom.resetPromptBtn.disabled = true;
  dom.copyPromptBtn.disabled = false;

  updateSaveBadge(true);

  // 画像プレビュー
  const imgUrl = `/api/image?subset=${encodeURIComponent(state.currentSubset)}&filename=${encodeURIComponent(state.selectedItem.filename)}`;
  dom.previewImage.src = imgUrl;
  dom.previewImage.style.display = 'block';
  dom.noImageSelected.style.display = 'none';
  dom.zoomImageBtn.style.display = 'flex';

  // メタデータ
  dom.imageMetaBar.style.display = 'flex';
  dom.metaResolution.textContent = state.selectedItem.width ? `${state.selectedItem.width} × ${state.selectedItem.height} px` : '-';
  dom.metaAspect.textContent = `${state.selectedItem.aspect_ratio} (${state.selectedItem.ratio_float})`;
  dom.metaFileSize.textContent = formatFileSize(state.selectedItem.file_size);
  
  if (state.selectedItem.has_tensor) {
    dom.metaTensorBadge.className = 'tensor-tag has';
    dom.metaTensorBadge.textContent = '生成済 (.pt)';
  } else {
    dom.metaTensorBadge.className = 'tensor-tag none';
    dom.metaTensorBadge.textContent = '未生成';
  }

  updateRealtimeCounts();
  highlightActiveCard();
  updateNavCounter();
}

function clearEditor() {
  state.selectedIndex = -1;
  state.selectedItem = null;
  state.isDirty = false;

  dom.currentFilename.textContent = '画像を選択してください';
  dom.currentExt.textContent = '-';
  dom.promptTextarea.value = '';
  dom.promptTextarea.disabled = true;
  dom.savePromptBtn.disabled = true;
  dom.resetPromptBtn.disabled = true;
  dom.copyPromptBtn.disabled = true;

  dom.previewImage.style.display = 'none';
  dom.noImageSelected.style.display = 'flex';
  dom.zoomImageBtn.style.display = 'none';
  dom.imageMetaBar.style.display = 'none';

  updateSaveBadge(true);
  updateRealtimeCounts();
  updateNavCounter();
}

function updateNavCounter() {
  if (state.selectedIndex === -1 || state.filteredItems.length === 0) {
    dom.currentPositionIndex.textContent = '- / -';
    dom.prevBtn.disabled = true;
    dom.nextBtn.disabled = true;
  } else {
    dom.currentPositionIndex.textContent = `${state.selectedIndex + 1} / ${state.filteredItems.length}`;
    dom.prevBtn.disabled = state.selectedIndex <= 0;
    dom.nextBtn.disabled = state.selectedIndex >= state.filteredItems.length - 1;
  }
}

function navigateImage(direction) {
  const newIndex = state.selectedIndex + direction;
  if (newIndex >= 0 && newIndex < state.filteredItems.length) {
    selectItem(newIndex);
  }
}

// ==========================================================================
// リアルタイム文字数・単語数カウント
// ==========================================================================
function updateRealtimeCounts() {
  const text = dom.promptTextarea.value;
  
  // 1. 全文字数（スペース含む）
  const totalChars = text.length;
  // 2. 空白除外文字数
  const noSpaceChars = text.replace(/\s+/g, '').length;
  // 3. 行数
  const lines = text ? text.split(/\r\n|\r|\n/).length : 0;
  // 4. 単語数 (簡易形態素・英単語ハイブリッド)
  const words = estimateWordCount(text);
  // 5. BERTトークン数の目安 (日本語BERT: 約1.2〜1.5文字で1サブワードトークン)
  const estTokens = estimateBertTokens(text);

  dom.countTotal.textContent = totalChars.toLocaleString();
  dom.countNoSpaces.textContent = noSpaceChars.toLocaleString();
  dom.countWords.textContent = words.toLocaleString();
  dom.countLines.textContent = lines.toLocaleString();
  dom.countTokens.textContent = estTokens.toLocaleString();

  // 256トークン超過時の警告色切替
  if (estTokens > 256) {
    dom.countTokens.style.color = '#ef4444';
    dom.tokenLimitBadge.style.color = '#ef4444';
  } else if (estTokens > 200) {
    dom.countTokens.style.color = '#f59e0b';
    dom.tokenLimitBadge.style.color = '#f59e0b';
  } else {
    dom.countTokens.style.color = '';
    dom.tokenLimitBadge.style.color = '';
  }
}

function estimateWordCount(text) {
  if (!text.trim()) return 0;
  // 漢字・カタカナ・英数字・ひらがなブロックにマッチ
  const matches = text.match(/[a-zA-Z0-9_\-]+|[\u4e00-\u9fff]+|[\u30a0-\u30ffー]+|[\u3040-\u309f]{2,}/g);
  return matches ? matches.length : 0;
}

function estimateBertTokens(text) {
  if (!text.trim()) return 0;
  // 日本語BERT (cl-tohoku) は、漢字やカタカナ、サブワード分割により
  // 平均して「文字数 × 0.75〜0.9」または単語分割数 + 特殊トークン([CLS], [SEP]) 程度になります。
  const charLen = text.replace(/\s+/g, '').length;
  const wordCount = estimateWordCount(text);
  // CLS, SEPの2トークンを加算し、目安値を算出
  return Math.min(512, Math.round(wordCount * 0.7 + charLen * 0.45) + 2);
}

function checkDirtyState() {
  const current = dom.promptTextarea.value;
  const isChanged = current !== state.originalPrompt;
  state.isDirty = isChanged;
  updateSaveBadge(!isChanged);
  dom.savePromptBtn.disabled = !isChanged;
  dom.resetPromptBtn.disabled = !isChanged;
}

function updateSaveBadge(isSaved) {
  if (isSaved) {
    dom.saveStatusBadge.className = 'status-badge saved';
    dom.saveStatusBadge.textContent = '保存済み';
  } else {
    dom.saveStatusBadge.className = 'status-badge unsaved';
    dom.saveStatusBadge.textContent = '未保存';
  }
}

// ==========================================================================
// プロンプト保存・リセット・コピー
// ==========================================================================
async function saveCurrentPrompt() {
  if (!state.selectedItem) return;

  const newPrompt = dom.promptTextarea.value;
  dom.savePromptBtn.disabled = true;
  dom.savePromptBtn.innerHTML = '<span>保存中...</span>';

  try {
    const res = await fetch('/api/save_prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        subset: state.currentSubset,
        base_name: state.selectedItem.base_name,
        prompt: newPrompt
      })
    });
    const data = await res.json();

    if (data.success) {
      state.originalPrompt = newPrompt;
      state.selectedItem.prompt = newPrompt;
      state.selectedItem.char_count = data.char_count;
      state.selectedItem.char_count_no_spaces = data.char_count_no_spaces;
      state.isDirty = false;

      updateSaveBadge(true);
      dom.resetPromptBtn.disabled = true;

      // ギャラリーカード上の文字数バッジも更新
      const card = document.getElementById(`card-${state.selectedItem.id}`);
      if (card) {
        const charBadge = card.querySelector('.card-char-badge');
        if (charBadge) charBadge.textContent = `${data.char_count}字`;
      }

      updateHeaderStats();
      showToast(`プロンプトを保存しました (${state.selectedItem.base_name}.txt)`, 'success');
    } else {
      throw new Error(data.error || '保存に失敗しました');
    }
  } catch (err) {
    showToast(`保存エラー: ${err.message}`, 'error');
  } finally {
    dom.savePromptBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
        <polyline points="17 21 17 13 7 13 7 21"></polyline>
        <polyline points="7 3 7 8 15 8"></polyline>
      </svg>
      <span>保存する</span>
    `;
    dom.savePromptBtn.disabled = !state.isDirty;
  }
}

function resetCurrentPrompt() {
  if (!state.selectedItem) return;
  dom.promptTextarea.value = state.originalPrompt;
  updateRealtimeCounts();
  checkDirtyState();
  showToast('編集内容を元に戻しました', 'info');
}

function copyCurrentPrompt() {
  const text = dom.promptTextarea.value;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    showToast('プロンプトをクリップボードにコピーしました', 'success');
  }).catch(() => {
    showToast('クリップボードへのコピーに失敗しました', 'error');
  });
}

// ==========================================================================
// 単語出現頻度・統計モーダル
// ==========================================================================
async function openWordStatsModal() {
  dom.wordStatsModal.style.display = 'flex';
  await fetchWordStats();
}

function closeWordStatsModal() {
  dom.wordStatsModal.style.display = 'none';
}

async function fetchWordStats() {
  dom.wordStatsTableBody.innerHTML = '<tr><td colspan="6" class="text-center">解析中...</td></tr>';
  dom.wordCloudContainer.innerHTML = '<div class="text-center" style="width: 100%;">単語頻度を集計中...</div>';

  const filterStopwords = dom.filterStopwordsCheck.checked;
  const minLen = dom.minWordLenSelect.value;
  const subset = state.currentSubset;

  try {
    const url = `/api/word_stats?subset=${encodeURIComponent(subset)}&min_length=${minLen}&filter_stopwords=${filterStopwords}`;
    const res = await fetch(url);
    const data = await res.json();
    state.wordStatsData = data;

    dom.statsTotalTokens.textContent = (data.total_words || 0).toLocaleString();
    dom.statsUniqueWords.textContent = (data.unique_words || 0).toLocaleString();
    dom.statsTotalDocs.textContent = (data.total_docs || 0).toLocaleString();
    dom.statsEngineName.textContent = data.tokenizer_engine === 'janome' ? 'Janome (形態素解析)' : '内蔵正規表現エンジン';

    renderWordStats();
  } catch (err) {
    showToast(`単語統計の取得に失敗しました: ${err.message}`, 'error');
  }
}

function renderWordStats() {
  if (!state.wordStatsData || !state.wordStatsData.words) return;

  const searchQuery = dom.wordSearchInput.value.trim().toLowerCase();
  const words = state.wordStatsData.words.filter(w => {
    if (!searchQuery) return true;
    return w.word.toLowerCase().includes(searchQuery);
  });

  // 1. テーブルの描画
  if (words.length === 0) {
    dom.wordStatsTableBody.innerHTML = '<tr><td colspan="6" class="text-center">一致する単語が見つかりませんでした</td></tr>';
  } else {
    dom.wordStatsTableBody.innerHTML = words.map((w, idx) => `
      <tr>
        <td style="color: var(--text-dim); font-family: var(--font-mono);">${idx + 1}</td>
        <td class="word-name-cell">${escapeHtml(w.word)}</td>
        <td><span class="count-badge">${w.count.toLocaleString()}</span> 回</td>
        <td>${w.doc_count.toLocaleString()} 枚</td>
        <td>
          <div class="ratio-bar-wrapper">
            <div class="ratio-progress-bg">
              <div class="ratio-progress-fill" style="width: ${w.doc_percent}%;"></div>
            </div>
            <span class="ratio-percent-text">${w.doc_percent}%</span>
          </div>
        </td>
        <td>
          <button class="btn-filter-word" onclick="applyWordFilter('${escapeHtml(w.word)}')">絞り込み</button>
        </td>
      </tr>
    `).join('');
  }

  // 2. ワードクラウドの描画
  if (words.length === 0) {
    dom.wordCloudContainer.innerHTML = '<div class="text-center" style="width: 100%;">表示できるタグがありません</div>';
  } else {
    // 上位120件をタグ表示
    const topWords = words.slice(0, 120);
    const maxCount = topWords.length > 0 ? topWords[0].count : 1;
    const minCount = topWords.length > 0 ? topWords[topWords.length - 1].count : 1;

    dom.wordCloudContainer.innerHTML = topWords.map(w => {
      // フォントサイズ: 0.75rem 〜 1.8rem
      const weight = maxCount > minCount ? (w.count - minCount) / (maxCount - minCount) : 0.5;
      const fontSize = 0.75 + weight * 1.05;
      const color = weight > 0.6 ? '#818cf8' : (weight > 0.3 ? '#38bdf8' : '#e2e8f0');

      return `
        <span class="cloud-tag" style="font-size: ${fontSize.toFixed(2)}rem; color: ${color};" onclick="applyWordFilter('${escapeHtml(w.word)}')">
          <strong>${escapeHtml(w.word)}</strong>
          <span class="cloud-tag-count">(${w.count})</span>
        </span>
      `;
    }).join('');
  }
}

// グローバルスコープに登録（HTML onclick用）
window.applyWordFilter = function(word) {
  state.activeWordFilter = word;
  dom.activeFilterWord.textContent = word;
  dom.activeWordFilter.style.display = 'flex';
  closeWordStatsModal();
  applyFiltersAndSort();
  showToast(`単語「${word}」で絞り込みました`, 'info');
};

function clearWordFilter() {
  state.activeWordFilter = null;
  dom.activeWordFilter.style.display = 'none';
  applyFiltersAndSort();
}

function exportWordStatsCsv() {
  if (!state.wordStatsData || !state.wordStatsData.words) return;
  const rows = [
    ['順位', '単語', '出現回数', '出現画像数', '出現割合(%)']
  ];
  state.wordStatsData.words.forEach((w, i) => {
    rows.push([i + 1, `"${w.word.replace(/"/g, '""')}"`, w.count, w.doc_count, w.doc_percent]);
  });

  const csvContent = '\uFEFF' + rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `word_stats_${state.currentSubset || 'all'}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ==========================================================================
// 一括単語置換モーダル
// ==========================================================================
function openBatchReplaceModal() {
  dom.batchReplaceModal.style.display = 'flex';
  dom.findWordInput.value = '';
  dom.replaceWordInput.value = '';
  dom.replacePreviewArea.style.display = 'none';
  dom.executeReplaceBtn.disabled = true;
}

function closeBatchReplaceModal() {
  dom.batchReplaceModal.style.display = 'none';
}

async function executeBatchReplace(previewOnly) {
  const findWord = dom.findWordInput.value.trim();
  const replaceWord = dom.replaceWordInput.value;
  const targetSubset = dom.replaceTargetSubset.value === 'current' ? state.currentSubset : 'all';

  if (!findWord) {
    showToast('検索する単語を入力してください', 'warning');
    return;
  }

  try {
    const res = await fetch('/api/batch_replace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        subset: targetSubset,
        find_word: findWord,
        replace_word: replaceWord,
        preview_only: previewOnly
      })
    });
    const data = await res.json();

    if (previewOnly) {
      dom.replacePreviewArea.style.display = 'flex';
      dom.replaceCount.textContent = data.modified_count;
      dom.executeReplaceBtn.disabled = data.modified_count === 0;

      if (data.modified_count === 0) {
        dom.replacePreviewList.innerHTML = '<div class="preview-item">一致するプロンプトは見つかりませんでした。</div>';
      } else {
        dom.replacePreviewList.innerHTML = data.changes.slice(0, 15).map(c => `
          <div class="preview-item">
            <strong>${escapeHtml(c.filename)}</strong> (${c.occurrences}箇所)<br>
            <span style="color: var(--danger); text-decoration: line-through;">${escapeHtml(findWord)}</span>
            &rarr;
            <span style="color: var(--success);">${escapeHtml(replaceWord)}</span>
          </div>
        `).join('') + (data.modified_count > 15 ? `<div style="text-align:center; color: var(--text-dim);">他 ${data.modified_count - 15} 件...</div>` : '');
      }
    } else {
      // 実行完了
      showToast(`${data.modified_count} 件のプロンプトを一括置換しました！`, 'success');
      closeBatchReplaceModal();
      // 再読み込み
      await switchSubset(state.currentSubset);
    }
  } catch (err) {
    showToast(`置換エラー: ${err.message}`, 'error');
  }
}

// ==========================================================================
// テンソル生成 (prepare_text.py)
// ==========================================================================
function openPrepareModal() {
  dom.prepareModal.style.display = 'flex';
  dom.prepareStatusText.textContent = '待機中';
  dom.prepareSpinner.style.display = 'none';
  dom.startPrepareBtn.disabled = false;
  pollPrepareStatus();
}

function closePrepareModal() {
  dom.prepareModal.style.display = 'none';
  if (state.preparePollingInterval) {
    clearInterval(state.preparePollingInterval);
    state.preparePollingInterval = null;
  }
}

async function startPrepareText() {
  dom.startPrepareBtn.disabled = true;
  dom.prepareSpinner.style.display = 'block';
  dom.prepareStatusText.textContent = '実行中...';

  try {
    const res = await fetch('/api/run_prepare_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subset: state.currentSubset })
    });
    const data = await res.json();
    if (!data.success) {
      throw new Error(data.error);
    }

    if (state.preparePollingInterval) clearInterval(state.preparePollingInterval);
    state.preparePollingInterval = setInterval(pollPrepareStatus, 1000);
  } catch (err) {
    showToast(`実行エラー: ${err.message}`, 'error');
    dom.startPrepareBtn.disabled = false;
    dom.prepareSpinner.style.display = 'none';
    dom.prepareStatusText.textContent = 'エラー';
  }
}

async function pollPrepareStatus() {
  try {
    const res = await fetch('/api/prepare_status');
    const data = await res.json();

    if (data.logs && data.logs.length > 0) {
      dom.prepareLogs.textContent = data.logs.join('\n');
      dom.prepareLogs.scrollTop = dom.prepareLogs.scrollHeight;
    }

    if (data.running) {
      dom.prepareStatusText.textContent = `実行中 (${data.subset})`;
      dom.prepareSpinner.style.display = 'block';
      dom.startPrepareBtn.disabled = true;
    } else {
      dom.prepareSpinner.style.display = 'none';
      dom.startPrepareBtn.disabled = false;
      if (data.returncode !== null) {
        if (data.returncode === 0) {
          dom.prepareStatusText.textContent = '完了 (成功)';
          dom.prepareStatusText.style.color = 'var(--success)';
        } else {
          dom.prepareStatusText.textContent = `終了 (コード: ${data.returncode})`;
          dom.prepareStatusText.style.color = 'var(--danger)';
        }
      }
    }
  } catch (e) {
    console.error(e);
  }
}

// ==========================================================================
// フルスクリーン画像ズーム
// ==========================================================================
function openFullscreenImage() {
  if (!state.selectedItem) return;
  const imgUrl = `/api/image?subset=${encodeURIComponent(state.currentSubset)}&filename=${encodeURIComponent(state.selectedItem.filename)}`;
  dom.fullscreenImage.src = imgUrl;
  dom.fullscreenModal.style.display = 'flex';
}

function closeFullscreenImage() {
  dom.fullscreenModal.style.display = 'none';
}

// ==========================================================================
// グローバルショートカット & リサイズ
// ==========================================================================
function handleGlobalKeydown(e) {
  // Ctrl + S: プロンプト保存
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
    e.preventDefault();
    if (state.isDirty) {
      saveCurrentPrompt();
    }
    return;
  }

  // テキストエリアにフォーカスがない場合の左右ナビゲーション
  const isInputFocused = document.activeElement === dom.promptTextarea ||
                         document.activeElement === dom.searchInput ||
                         document.activeElement === dom.wordSearchInput;

  if (!isInputFocused) {
    if (e.key === 'ArrowLeft' || e.key === 'k') {
      navigateImage(-1);
    } else if (e.key === 'ArrowRight' || e.key === 'j') {
      navigateImage(1);
    }
  } else if ((e.ctrlKey || e.metaKey) && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
    e.preventDefault();
    navigateImage(e.key === 'ArrowLeft' ? -1 : 1);
  }

  // Escape: モーダルを閉じる
  if (e.key === 'Escape') {
    if (dom.fullscreenModal.style.display === 'flex') closeFullscreenImage();
    if (dom.wordStatsModal.style.display === 'flex') closeWordStatsModal();
    if (dom.batchReplaceModal.style.display === 'flex') closeBatchReplaceModal();
    if (dom.prepareModal.style.display === 'flex') closePrepareModal();
  }
}

function initPaneResizer() {
  let isResizing = false;
  const resizer = dom.paneResizer;
  const galleryPane = document.querySelector('.gallery-pane');

  resizer.addEventListener('mousedown', (e) => {
    isResizing = true;
    resizer.classList.add('resizing');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  window.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    const containerWidth = document.querySelector('.main-layout').offsetWidth;
    const newWidth = Math.max(340, Math.min(e.clientX, containerWidth - 380));
    galleryPane.style.flex = `0 0 ${newWidth}px`;
    galleryPane.style.maxWidth = `${newWidth}px`;
  });

  window.addEventListener('mouseup', () => {
    if (isResizing) {
      isResizing = false;
      resizer.classList.remove('resizing');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }
  });
}

// ==========================================================================
// ユーティリティ
// ==========================================================================
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  dom.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}

function formatFileSize(bytes) {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>"']/g, m => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[m]);
}
