"""Text preprocessing pipeline for topic modelling.

Provides language-aware stopword removal, text cleaning (URLs, mentions,
emojis, numbers), and a corpus-level pipeline that prepares raw text for
BERTopic.  Cleaning is intentionally *light* for the embedding step —
stopwords are only removed in the CountVectorizer (representation layer),
not from the text the transformer model reads.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

# ── Stopword lists (compact, one string per language) ─────────────────────────
# These cover articles, prepositions, pronouns, common verbs, conjunctions,
# determiners, and high-frequency filler words.  Users can extend via the
# custom stopwords field.

_STOP = {
    "en": (
        "a about above after again against all am an and any are aren't as at "
        "be because been before being below between both but by can can't cannot "
        "could couldn't did didn't do does doesn't doing don't down during each "
        "few for from further get got had hadn't has hasn't have haven't having "
        "he her here hers herself him himself his how i i'd i'll i'm i've if in "
        "into is isn't it it's its itself just let's like me more most mustn't "
        "my myself no nor not now of off on once only or other ought our ours "
        "ourselves out over own really same shan't she should shouldn't so some "
        "such than that the their theirs them themselves then there these they "
        "this those through to too under until up upon us very was wasn't we "
        "we'd we'll we're we've were weren't what when where which while who "
        "whom why will with won't would wouldn't you your yours yourself "
        "yourselves also always another anything back been being come could "
        "even every first get give go going good got great has have her here "
        "him how its just know last like long look made make many may more "
        "most much must new next now old one only other over own part right "
        "same say see she some still such take tell than that the them then "
        "there these think this time two use very want way well work would "
        "year yes yet"
    ),
    "pt": (
        "a à ao aos aquela aquelas aquele aqueles aquilo as assim até bem bom "
        "cada com como contra da das de dela delas dele deles depois desde dessa "
        "dessas desse desses desta destas deste destes deve do dos e ela elas "
        "ele eles em entre era essa essas esse esses esta estas este estes "
        "estava estou eu fazer fez foi for foram fosse há isso isto já lá lhe "
        "lhes lo mais mas me menos meu minha meus minhas muito na nada não nas "
        "nem nessa neste no nos nós nossa nossas nosso nossos num numa o os ou "
        "outra outras outro outros para pela pelas pelo pelos per pode podem "
        "por porém porque qual quando que quem se sem ser será seu seus si sido "
        "só sobre sua suas também tem tendo tenho ter teu teus toda todas todo "
        "todos tu tua tuas tudo um uma umas uns vai vamos você vocês vos vossa "
        "vosso ainda algo alguém alguma algumas algum alguns ali antes aqui "
        "atrás bastante cá certo coisa coisas demais depois então era essa "
        "estar este forma grande hoje lá logo melhor menos mesma mesmo muita "
        "muitas muitos nenhum nenhuma ninguém noite nunca onde outra parte "
        "pois porque própria próprio quais qualquer quanto quase sempre senão "
        "somente talvez tanto tipo toda vezes agora aquele após através coisa "
        "conta dentro dia então esses estar estas este forma gente grande "
        "maior meio mesmo modo mundo nada nesta neste nova novo onde outra "
        "outro parte pessoa pouco primeira primeiro pode pra quem quer sabe "
        "ser sobre toda vamos vez vida"
    ),
    "es": (
        "a al algo algunas alguno algunos algún ante antes aquí aquel aquella "
        "aquellas aquellos así aunque bien cada casi como con contra cosas "
        "creo cual cuando da de del desde donde dos el ella ellas ellos en "
        "entre era esa esas ese esos esta estaba estado están estar estas "
        "este esto estos fue fueron ha había hacer hacia han hasta hay he "
        "hemos hoy la las le les lo los luego más me mejor menos mi mí "
        "mientras mis mismo mucho muy nada ni ninguno no nos nosotros nuestro "
        "nuestros nueva nuevo o otra otras otro otros para pero poco por "
        "porque puede pueden pues que qué quien quién se sea según ser si "
        "sí sin sino sobre somos son soy su sus también tan tanto te tengo "
        "ti tiene tiempo toda todavía todo todos tres tu tú tus tuvo un una "
        "uno unos usted ustedes va vamos verdad vez ya yo"
    ),
    "fr": (
        "à ai aie aient aies ait alors après au aucun aura aurait aussi autre "
        "autres aux avait avant avec avoir ayant bien bon ça car ce cela celle "
        "celui ces cet cette chez comme comment dans de del des depuis devrait "
        "dit dois doit donc dont du elle elles en encore entre est et eu eux "
        "fait faire faut fois il ils ici jamais je jour juste la là le les "
        "leur leurs lui ma mais me même mes moi moins mon ne ni nom non nos "
        "notre nous on ont or ou où par parce pas pendant peut plus plusieurs "
        "pour pourquoi puis qu que quel quelle quelque qui quoi rien sa sans "
        "se sera ses si soi sommes son sont sous suis sur ta te temps tes toi "
        "ton toujours tous tout toute toutes très trop tu un une vos votre "
        "vous vraiment y"
    ),
    "de": (
        "ab aber ach alle allein allem allen aller allerdings alles also am an "
        "ander andere anderem anderen anderer anderes anderm andern anderr "
        "anders als auch auf aus bei beim bereits besonders bin bis bisher bist "
        "da dabei dadurch dafür dagegen daher dahin damals damit danach daneben "
        "dann daran darauf daraus darf darfst darin darum darunter darüber das "
        "dass dazu dein deine deinem deinen deiner dem den denn der des dessen "
        "dich die dies diese dieselbe dieselben diesem diesen dieser dieses dir "
        "doch dort drei du durch dürfen ein eine einem einen einer einige "
        "einigen einiger einiges einmal er es etwas euch euer eure eurem euren "
        "eurer für ganz gar gegen gehen geht gemacht gerade gern getan gewesen "
        "gibt groß große großen großer großes gut haben halt hat hatte hätte "
        "habe hast hier hin hinter ich ihm ihn ihnen ihr ihre ihrem ihren ihrer "
        "immer in indem ins irgend ist ja jede jedem jeden jeder jedes jedoch "
        "jemals jene jenem jenen jener jenes jetzt kann kannst kein keine "
        "keinem keinen keiner könnte machen mag manche manchem manchen mancher "
        "manches man mehr mein meine meinem meinen meiner mir mit möchte muß "
        "muss müssen nach nachdem nachher nämlich natürlich neben nein nicht "
        "nichts noch nun nur ob oder ohne schon sehr seid sein seine seinem "
        "seinen seiner seit sich sicher sie sind so sogar solch solche solchem "
        "solchen solcher soll sollen sollte sollten sondern sonst soweit sowie "
        "über um und uns unser unsere unserem unseren unserer unter viel "
        "vielleicht vom von vor während wann warum was weder weil weit welch "
        "welche welchem welchen welcher wenn wer werde werden wessen wie wieder "
        "will wir wird wo wohl wollen worden würde würden zu zum zur zusammen "
        "zwischen"
    ),
    "it": (
        "a abbia abbiamo abbiano abbiate ad agl agli ai al alla alle allo "
        "ancora anche avemmo avendo avere avesse avessero avessi avessimo "
        "aveste avesti avete aveva avevamo avevano avevate avevi avevo avrai "
        "avranno avrebbe avrebbero avrei avremmo avremo avreste avresti avrete "
        "avrà avrò avuta avute avuti avuto c che chi ci co col come con contro "
        "cui da dagl dagli dai dal dall dalla dalle dallo dei del dell della "
        "delle dello di dopo dove e ebbe ebbero ebbi ed era erano eravamo "
        "eravate eri ero fa facciamo fai fanno fare farà farebbe fece fino fra "
        "fu fui fummo furono gli ha hai hanno ho i il in io l la le lei li lo "
        "loro lui ma me mi mia mie miei mio molto ne negl negli nei nel nell "
        "nella nelle nello no noi non nostra nostre nostri nostro o per perché "
        "più prima primo può qualche quale quali quando quanto quei quel "
        "quella quelle quelli quello questa queste questi questo sarai "
        "saranno sarebbe sarebbero sarei saremmo saremo sareste saresti "
        "sarete sarà sarò se sei si sia siamo siano siate siete sono sta "
        "stai stanno stato stava stesso sto su sua sue sugl sugli sui sul "
        "sull sulla sulle sullo suo suoi ti tra tu tua tue tuo tuoi tutti "
        "tutto un una uno vi voi vostra vostre vostri vostro"
    ),
    "nl": (
        "aan al alles als altijd ander andere anders ben bij daar dan dat de "
        "der deze die dit doch doen door dus een eens eigen einde er even "
        "gebruik geen geweest haar had heb hebben heeft hem het hier hij hoe "
        "hun ieder iemand iets ik in is ja je kan kon kunnen laat lang later "
        "laten maar mag maken me meer men met mij mijn moest moet mogen na "
        "naar niet niets nog nou nu of om omdat ons ook op over reeds reed "
        "samen sinds slechts slecht steeds te tegen toch toen tot twee u uit "
        "uw van veel voor want was wat we wel welk werd wie wij wil worden "
        "wordt zo zou zijn"
    ),
    "tr": (
        "acaba acep ait ama ancak arada aslında az bana bazı ben beni benim "
        "bir birçok birkaç biz bize bizi bizim böyle bu buna bunda bundan "
        "bunlar bunları bunların bunun burada çok çünkü da daha de değil "
        "defa değil diye diğer dolayı ederek en fazla fakat gibi göre hala "
        "hangi hatta hem henüz hep hepsi her herhangi herkes hiç hiçbir "
        "için ise işte kadar karşı kendi kendine kim kime kimse nasıl ne "
        "neden nedir nere nerede nereye niçin niye olan olarak oldu olmak "
        "olması olmaz olsa olup olur oluyor ona ondan onlar onları onların "
        "onu onun orada oysa öyle pek şey şu şuna şundan şunlar şunu ta "
        "tam tamam tüm üzere var ve veya ya yani yapılan yapılması yapıyor "
        "yoksa zaten"
    ),
}

# Social-media-specific noise tokens (cross-language)
_SOCIAL_MEDIA = (
    "rt via http https www com amp lol lmao omg smh tbh imo imho fyi brb "
    "dm dms pic pics link bio follow retweet tweet post share like comment "
    "subscribe click hashtag thread"
)

# ── Build sets ────────────────────────────────────────────────────────────────

STOPWORDS: dict[str, set[str]] = {
    lang: set(words.split()) for lang, words in _STOP.items()
}
SOCIAL_STOPS: set[str] = set(_SOCIAL_MEDIA.split())

LANGUAGE_NAMES: dict[str, str] = {
    "auto": "Auto-detect",
    "en":   "English",
    "pt":   "Portuguese",
    "es":   "Spanish",
    "fr":   "French",
    "de":   "German",
    "it":   "Italian",
    "nl":   "Dutch",
    "tr":   "Turkish",
    "none": "No stopwords",
}

# ── Regex patterns ────────────────────────────────────────────────────────────
_URL_RE      = re.compile(r"https?://\S+|www\.\S+", re.I)
_MENTION_RE  = re.compile(r"@[\w.]+")
_HASHTAG_RE  = re.compile(r"#(\w+)")          # capture group keeps the word
_EMOJI_RE    = re.compile(
    "[\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"    # symbols & pictographs
    "\U0001F680-\U0001F6FF"    # transport & map
    "\U0001F900-\U0001F9FF"    # supplemental
    "\U0001FA00-\U0001FA6F"    # chess symbols
    "\U0001FA70-\U0001FAFF"    # symbols extended-A
    "\U00002702-\U000027B0"    # dingbats
    "\U0000FE00-\U0000FE0F"    # variation selectors
    "\U0000200D"               # zero width joiner
    "\U00002600-\U000026FF"    # misc symbols
    "]+", re.U)
_NUMBER_RE   = re.compile(r"\b\d+\b")
_MULTI_SPACE = re.compile(r"\s{2,}")


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def detect_language(texts: list[str], n_sample: int = 200) -> str:
    """Detect the dominant language in *texts* using stopword frequency.

    Returns a 2-letter language code (e.g. 'en', 'pt').  Falls back to 'en'
    if detection is inconclusive.
    """
    sample = texts[:n_sample]
    blob = " ".join(sample).lower()
    tokens = set(re.findall(r"\b\w{2,}\b", blob))

    best_lang, best_score = "en", 0
    for lang, stops in STOPWORDS.items():
        score = len(tokens & stops)
        if score > best_score:
            best_score = score
            best_lang = lang
    return best_lang


def clean_text(
    text: str,
    *,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    strip_hashtag_symbol: bool = True,
    remove_emojis: bool = True,
    remove_numbers: bool = True,
    lowercase: bool = True,
) -> str:
    """Apply light cleaning to a single text string.

    This is intended for the embedding step — stopwords are NOT removed here
    (the transformer model uses them for context).
    """
    if not text:
        return ""
    if remove_urls:
        text = _URL_RE.sub(" ", text)
    if remove_mentions:
        text = _MENTION_RE.sub(" ", text)
    if strip_hashtag_symbol:
        text = _HASHTAG_RE.sub(r"\1", text)     # keep word, drop #
    if remove_emojis:
        text = _EMOJI_RE.sub(" ", text)
    if remove_numbers:
        text = _NUMBER_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    # Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def build_stopword_set(
    language: str,
    custom_stopwords: Optional[list[str]] = None,
    include_social: bool = True,
) -> list[str]:
    """Build a combined stopword list for the CountVectorizer.

    Merges the built-in list for *language*, social-media noise tokens,
    and any user-supplied *custom_stopwords*.
    """
    stops: set[str] = set()
    if language in STOPWORDS:
        stops |= STOPWORDS[language]
    if include_social:
        stops |= SOCIAL_STOPS
    if custom_stopwords:
        for w in custom_stopwords:
            w = w.strip().lower()
            if w:
                stops.add(w)
    return sorted(stops)


def preprocess_corpus(
    texts: list[str],
    *,
    language: str = "auto",
    remove_urls: bool = True,
    remove_mentions: bool = True,
    strip_hashtag_symbol: bool = True,
    remove_emojis: bool = True,
    remove_numbers: bool = True,
    lowercase: bool = True,
    min_token_length: int = 3,
    min_doc_words: int = 5,
    custom_stopwords: Optional[list[str]] = None,
) -> tuple[list[str], str, list[str], dict]:
    """Full preprocessing pipeline.

    Returns
    -------
    cleaned_texts : list[str]
        Lightly cleaned texts (for embedding).
    detected_lang : str
        The language code used.
    stopword_list : list[str]
        Combined stopword list (for the CountVectorizer).
    stats : dict
        Cleaning statistics for the UI.
    """
    # 1. Detect language
    if language == "auto":
        detected_lang = detect_language(texts)
    else:
        detected_lang = language

    # 2. Light cleaning for embeddings
    cleaned = [
        clean_text(
            t,
            remove_urls=remove_urls,
            remove_mentions=remove_mentions,
            strip_hashtag_symbol=strip_hashtag_symbol,
            remove_emojis=remove_emojis,
            remove_numbers=remove_numbers,
            lowercase=lowercase,
        )
        for t in texts
    ]

    # 3. Filter short documents
    original_count = len(cleaned)
    kept = [(i, t) for i, t in enumerate(cleaned)
            if len(t.split()) >= min_doc_words and t.strip()]
    indices = [i for i, _ in kept]
    cleaned = [t for _, t in kept]
    removed_count = original_count - len(cleaned)

    # 4. Build stopword list for vectorizer
    stopword_list = build_stopword_set(
        detected_lang, custom_stopwords, include_social=True)

    stats = {
        "language":        LANGUAGE_NAMES.get(detected_lang, detected_lang),
        "original_count":  original_count,
        "cleaned_count":   len(cleaned),
        "removed_short":   removed_count,
        "stopwords_count": len(stopword_list),
        "indices":         indices,
    }
    return cleaned, detected_lang, stopword_list, stats


def preview_cleaning(
    texts: list[str],
    *,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    strip_hashtag_symbol: bool = True,
    remove_emojis: bool = True,
    remove_numbers: bool = True,
    lowercase: bool = True,
    n: int = 5,
) -> list[tuple[str, str]]:
    """Return (before, after) pairs for *n* sample texts."""
    import random
    sample = random.sample(texts, min(n, len(texts)))
    pairs = []
    for t in sample:
        cleaned = clean_text(
            t,
            remove_urls=remove_urls,
            remove_mentions=remove_mentions,
            strip_hashtag_symbol=strip_hashtag_symbol,
            remove_emojis=remove_emojis,
            remove_numbers=remove_numbers,
            lowercase=lowercase,
        )
        pairs.append((t, cleaned))
    return pairs
