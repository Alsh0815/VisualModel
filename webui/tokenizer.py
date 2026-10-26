import re
from collections import Counter
from typing import List, Dict

# 一般的な日本語の助詞・助動詞・接続詞・活用語尾などのストップワード
JAPANESE_STOPWORDS = {
    "の", "に", "は", "を", "た", "で", "て", "と", "し", "れ", "さ", "ある", "いる", "も", "する", "から",
    "な", "こと", "として", "い", "や", "れる", "など", "なっ", "ない", "この", "ため", "その", "あっ",
    "よう", "また", "もの", "という", "あり", "これ", "それ", "ここ", "そこ", "これら", "それら",
    "なり", "だ", "です", "ます", "でした", "ました", "ば", "へ", "か", "よ", "ね", "わ",
    "が", "お", "ご", "より", "なら", "ぬ", "ず", "たり", "ながら", "つつ", "でも", "しか",
    "られ", "させ", "せる", "ている", "てある", "ておく", "ていく", "てくる", "ようだ", "らしい",
    "そう", "たい", "たがる", "ないで", "なくて", "ず", "ぬ", "き", "く", "け", "こ",
    "、", "。", "！", "？", "「", "」", "（", "）", "【", "】", "…", "・", "―", "〜",
    ",", ".", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}", ":", ";", "/", "-", "~",
    "\n", "\r", "\t", " "
}

# Janomeの動的インポート試行
_janome_tokenizer = None
try:
    from janome.tokenizer import Tokenizer
    _janome_tokenizer = Tokenizer()
except Exception:
    _janome_tokenizer = None


def is_stopword(word: str) -> bool:
    """単語がストップワードまたは不要な記号・空白であるかを判定"""
    w = word.strip()
    if not w:
        return True
    if w in JAPANESE_STOPWORDS:
        return True
    if len(w) == 1 and not re.match(r'[a-zA-Z0-9\u4e00-\u9fff]', w):
        return True
    return False


def tokenize_text(text: str, use_janome: bool = True) -> List[str]:
    """
    テキストを単語（トークン）リストに分割する。
    Janomeが使用可能な場合は形態素解析を行い、名詞・動詞・形容詞等の原形を取得。
    使用できない場合は高精度なフォールバックを実施。
    """
    if not text or not text.strip():
        return []

    global _janome_tokenizer
    if use_janome and _janome_tokenizer is not None:
        try:
            tokens = []
            for token in _janome_tokenizer.tokenize(text):
                part_of_speech = token.part_of_speech.split(',')[0]
                # 自立語（名詞、動詞、形容詞、カスタム名詞、未知語）のみを抽出
                if part_of_speech in ('名詞', '動詞', '形容詞', '副詞', 'カスタム名詞', '未知語'):
                    base_form = token.base_form
                    if base_form == '*':
                        base_form = token.surface
                    if not is_stopword(base_form):
                        tokens.append(base_form)
            return tokens
        except Exception:
            pass

    # フォールバック: 精密なハイブリッドトークナイザー
    tokens = []

    # カンマ区切りのタグ風フレーズ処理
    if ',' in text:
        parts = [p.strip() for p in text.split(',') if p.strip()]
        for p in parts:
            if re.match(r'^[a-zA-Z0-9_\-\s]+$', p) and len(p) <= 40:
                tokens.append(p.lower())
                continue
            tokens.extend(_extract_meaningful_words(p))
    else:
        tokens.extend(_extract_meaningful_words(text))

    return tokens


def _extract_meaningful_words(text: str) -> List[str]:
    """
    意味を持つ単語（漢字熟語、カタカナ語、英単語、漢字＋送り仮名の複合語）を高精度に抽出
    ひらがなのみの助詞・接続詞塊は除外
    """
    tokens = []
    # 1. カタカナ語（2文字以上）
    # 2. 漢字熟語（1文字以上）または 漢字+送り仮名（例: 愛らしい, 描いた, 覗いて, 身につけて）
    # 3. 英数字・アンダースコア（1文字以上）
    pattern = re.compile(
        r'[\u30a0-\u30ffー]{2,}'                               # カタカナ語 (リボン, メイド服のメイド, etc.)
        r'|[\u4e00-\u9fff]{1,8}(?:[\u3040-\u309f]{1,4})?'     # 漢字または漢字＋送り仮名
        r'|[a-zA-Z0-9_\-]+'                                    # 英数字
    )

    matches = pattern.findall(text)
    for word in matches:
        word = word.strip()
        if not word or is_stopword(word):
            continue

        # 送り仮名付き活用語の末尾トリミング（例: "描いた" -> "描く/描い", "載って" -> "載る", 等の簡易正規化）
        # 活用語尾のノイズを低減
        for suffix in ("ている", "てある", "ていること", "している", "を描いた", "になった", "とした", "が見える", "を持つ"):
            if word.endswith(suffix) and len(word) > len(suffix):
                word = word[:-len(suffix)]
                break

        if not is_stopword(word) and len(word) >= 1:
            if re.match(r'^[a-zA-Z0-9_\-]+$', word):
                tokens.append(word.lower())
            else:
                tokens.append(word)

    return tokens


def calculate_word_frequencies(
    dataset_records: List[Dict[str, str]],
    filter_stopwords: bool = True,
    min_length: int = 1
) -> Dict:
    """
    データセット全体のプロンプトから単語の出現頻度を算出する。
    """
    total_words = 0
    word_counter = Counter()
    doc_counter = Counter()
    word_to_files = {}

    for record in dataset_records:
        prompt = record.get("prompt", "")
        file_name = record.get("filename", "")
        if not prompt:
            continue

        tokens = tokenize_text(prompt)
        seen_in_doc = set()

        for token in tokens:
            if filter_stopwords and is_stopword(token):
                continue
            if len(token) < min_length:
                continue

            total_words += 1
            word_counter[token] += 1

            if token not in seen_in_doc:
                doc_counter[token] += 1
                seen_in_doc.add(token)
                if token not in word_to_files:
                    word_to_files[token] = []
                if len(word_to_files[token]) < 50:
                    word_to_files[token].append(file_name)

    stats = []
    total_docs = len(dataset_records)

    for word, count in word_counter.most_common():
        doc_count = doc_counter[word]
        stats.append({
            "word": word,
            "count": count,
            "doc_count": doc_count,
            "doc_percent": round((doc_count / total_docs * 100), 1) if total_docs > 0 else 0,
            "files": word_to_files.get(word, [])
        })

    return {
        "total_words": total_words,
        "unique_words": len(stats),
        "total_docs": total_docs,
        "tokenizer_engine": "janome" if _janome_tokenizer is not None else "built-in regex (optimized)",
        "words": stats
    }
