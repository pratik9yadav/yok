# Romanization variants for shop-call words (Hindi and Marathi)

`romanization_variants.json` lists, for words and short phrases that shopkeepers in Pune say on a
phone call, the Latin-letter spellings that real people were seen to use, with counts and the
source of each count. It is meant for making the romanized `said` text of training examples look
like real typing (nahi / nhi / nahin, ahe / aahe, rupaye / rs, ...), not for scoring.

## Summary

- Word list: 308 rows in `word_list.tsv` (Devanagari word or phrase, language, gloss, category, seed spellings). Rows with the same Devanagari form in Hindi and Marathi are merged, giving **288 entries** (170 Hindi, 98 Marathi, 20 both).
- **202 of 288 entries (70%) have 2 or more attested spellings**; 63 have exactly one; 23 have none.
  - Hindi (incl. shared): 145 of 190 with 2+ spellings (76%), 10 with none.
  - Marathi (incl. shared): 75 of 118 with 2+ spellings (64%), 13 with none.
- 1602 (spelling, source) rows in total, 1036 distinct (entry, spelling) pairs (after the independent check removed 110 wrong pairs; see `check_report.json`).

| source | entries with counts | spelling rows | total count |
|---|---:|---:|---:|
| comi-lingua-tn | 166 | 627 | 107,457 |
| dakshina-hi-lexicon | 159 | 246 | 692 |
| dakshina-hi-sentences | 141 | 335 | 30,844 |
| dakshina-mr-lexicon | 99 | 231 | 418 |
| dakshina-mr-sentences | 90 | 163 | 7,566 |

By category (entries with 2+ spellings / all):

alternative 9/9, availability 22/31, delivery 13/20, filler 31/45, function 17/18, loanword 12/26, number 16/23, pickup 19/27, price 21/26, refusal 7/14, time 26/32, unit 9/17

## Examples (weights as used for sampling)

| word | lang | meaning | top spellings |
|---|---|---|---|
| नहीं | hi | no / not | nahi 56%, nhi 24%, nahin 13%, ni 2%, nai 2%, nehi 1% |
| नहीं है | hi | is not there | nahi hai 54%, nhi hai 16%, nahin hai 13%, nhi h 6%, nahi h 3%, nahi hain 1% |
| है | hi | is (there) | hai 92%, h 7%, ha 1%, hain <1%, hay <1%, hei <1% |
| हैं | hi | are (there) | hain 74%, hai 14%, h 9%, haim 1%, ha 1%, hn <1% |
| हाँ | hi | yes | haan 38%, han 23%, ha 20%, haa 16%, hn 4% |
| जी | hi | ji (polite yes) | ji 98%, jii 2%, gi <1%, jiii <1%, je <1%, ju <1% |
| अभी | hi | right now | abhi 94%, avi 2%, abi 2%, aabhi 1%, abbi <1%, abh <1% |
| मिलेगा | hi | (you) will get / available | milega 92%, milegaa 5%, mile ga 1%, meliga 1% |
| खत्म | hi | finished / out of stock | khatam 64%, khatm 22%, khtm 7%, khtam 4%, khatma 2%, katam 1% |
| स्टॉक | hi+mr | stock | stock 94%, stok 3%, stauck 1%, stauk 1%, stocks 1% |
| माल | hi+mr | goods / stock | maal 83%, mal 17% |
| रुपये | hi+mr | rupees | rupaye 37%, rs 30%, rupees 17%, rupay 5%, rupye 4%, rupae 3% |
| रुपया | hi | rupee | rupaya 50%, rupya 38%, roopaya 6%, rupayaa 6% |
| कितना | hi | how much | kitna 95%, kitana 5%, kitnaa 1% |
| ठीक है | hi | okay | thik hai 65%, theek hai 23%, thik h 3%, theek h 3%, thik hain 2%, thk hai 2% |
| अच्छा | hi | ok / good | achha 41%, accha 36%, acha 11%, achcha 5%, achchha 2%, achchhaa 2% |
| भाई | hi | brother | bhai 98%, bahi 1%, bhaai 1%, vai <1%, bhya <1%, bhae <1% |
| भैया | hi | brother (polite) | bhaiya 89%, bhiya 5%, bhaya 2%, bhaiyya 1%, bhaiyaa 1%, bhyya 1% |
| बोलिए | hi | go ahead, tell me | boliye 91%, boliae 4%, boliy 4% |
| बताइए | hi | please tell | bataiye 73%, btaiye 11%, btaye 4%, bataeye 4%, bataye 4%, bataaiye 1% |
| भेजता | hi | (I) send | bhejta 63%, bhejata 18%, bhejataa 9%, bhejtaa 9% |
| डिलीवरी | hi+mr | delivery | delivery 100% |
| घर | hi | home | ghar 93%, ghr 6%, gher 1% |
| कल | hi | tomorrow / yesterday | kal 88%, kl 10%, cal 1%, kall 1% |
| आज | hi | today | aaj 89%, aj 9%, ajj 2%, ajh <1% |
| ज़्यादा | hi | more | jyada 66%, zyada 13%, jada 9%, jayda 2%, jyda 2%, jayada 2% |
| मीटर | hi+mr | metre | meter 56%, metre 11%, mitar 11%, meetar 10%, miter 4%, meeter 2% |
| किलो | hi+mr | kilo | kilo 49%, kg 43%, kelo 2%, keelo 1%, keeloo 1%, kiilo 1% |
| लीटर | hi+mr | litre | liter 52%, litre 23%, littre 4%, leetar 4%, letar 4%, leter 4% |
| पीस | hi+mr | piece | piece 50%, pees 26%, peice 24% |
| जोड़ी | hi+mr | pair | jodi 75%, jodee 5%, jori 3%, jodie 3%, jodii 3%, jodiie 3% |
| दूसरा | hi | another / other | dusra 74%, doosra 11%, dusara 9%, dusraa 3%, dusaraa 2% |
| वाला | hi | the one (with) | wala 78%, vala 19%, bala 1%, waala 1%, wla 1%, vaala <1% |
| बाद में | hi | later | baad men 25%, baad me 24%, baad mein 23%, bad me 13%, bad mein 6%, baad mai 3% |
| जल्दी | hi | soon / quickly | jaldi 80%, jldi 9%, jald 8%, jld 3%, jdi 1% |
| कंपनी | hi+mr | company / brand | company 89%, kampani 5%, kampni 2%, kampanee 1%, campany 1%, compani 1% |
| ओरिजिनल | hi+mr | original | original 100% |
| लोकल | hi+mr | local | local 86%, lokal 7%, loocal 4%, lookal 4% |
| आहे | mr | is (there) | ahe 66%, aahe 34%, aahai <1%, aaahe <1%, aae <1%, aaeh <1% |
| आहेत | mr | are (there) | ahet 64%, aahet 34%, aheta 1%, aaheta <1%, ahhet <1% |
| नाही | mr | no / not | nahi 100%, naahi <1% |
| हो | mr | yes | ho 62%, hoe 8%, hoo 8%, hooe 8%, hoowe 8%, howe 8% |
| मिळेल | mr | will get / available | milel 100% |
| मिळणार | mr | will be available | milanar 60%, milnar 20%, milnaar 20% |
| उद्या | mr | tomorrow | udya 100% |
| परवा | mr | day after tomorrow | parava 50%, parva 50% |
| किती | mr | how much | kiti 93%, kitee 7% |
| पाठवतो | mr | (I) will send | pathavato 50%, pathvato 50% |
| पाठवू | mr | (we) shall send | pathavu 50%, pathvoo 25%, pathvu 25% |
| घरी | mr | at home | ghari 90%, gharee 10% |
| घरपोच | mr | home delivery | gharpoch 75%, gharapoch 25% |
| इथे | mr | here | ithe 91%, ethe 9% |
| आत्ता | mr | right now | aatta 74%, atta 26% |
| फक्त | mr | only | fakta 39%, fakt 26%, phakt 25%, phakta 8%, faqt 3% |
| जास्त | mr | more | jasta 53%, jast 44%, jaast 2%, jasht 1% |
| दुसरा | mr | another (masc.) | dusara 49%, dusra 46%, dursa 4% |
| पाच | mr | five | pach 62%, paach 35%, panch 3% |
| अडीच | mr | two and a half | adich 50%, adeech 17%, adhich 17%, adicha 17% |
| भाऊ | mr | brother | bhau 95%, bhauu 5% |
| दादा | mr | elder brother / sir | dada 87%, daada 6%, daadaa 6% |
| ताई | mr | sister / madam | tai 44%, taaee 11%, tae 11%, taee 11%, taiee 11%, taii 11% |

## Entries with one or no attested spelling

One spelling only (usually because the word is rare in both sources, or every writer agreed):

मिलेगी (milegi), सेम (same), बोलो (bolo), सुनिए (suniye), कितने का (kitne ka), प्राइस (price), जीएसटी (gst), कैश (cash), डिस्काउंट (discount), फिक्स (fix), दो (do), सौ (sau), डेढ़ (dedh), पैकेट (packet), डिब्बा (dibba), डब्बा (dabba), बॉक्स (box), दर्जन (darjan), इंच (inch), भेज (bhej), भेजेंगे (bhejenge), डिलीवरी (delivery), फ्री (free), ले जाना (le jana), हफ्ते (hafte), थोड़ी देर (thodi der), ओरिजिनल (original), डुप्लीकेट (duplicate), साइज़ (size), क्वालिटी (quality), वारंटी (warranty), बिल्कुल (bilkul), मिल (mil), बिक (bik), होम (home), आहे का (aahe ka), मिळेल (milel), मिळत नाही (milat nahi), संपले (sampale), शिल्लक (shillak), उद्या (udya), नंतर (nantar), लगेच (lagech), दर (dar), दुकानात (dukanat), घ्या (ghya), देतो (deto), सांगा (sanga), सांगतो (sangato), दहा (daha), पन्नास (pannas), शंभर (shambhar), साडे (sade), डबा (daba), साहेब (saheb), बरं (bar), नक्की (nakki), जरा (jara), थांबा (thamba), महाग (mahag), काय (kay), तुम्ही (tumhi), मिळत (milat)

No attested spelling:

- जी हाँ [hi] yes (polite) - build from parts: जी (ji) + हाँ (haan)
- बिक गया [hi] sold out - build from parts: बिक (bik) + गया (gaya)
- भेज दूंगा [hi] I will send - build from parts: भेज (bhej) + दूंगा (dunga)
- भेज देंगे [hi] (we) will send - build from parts: भेज (bhej) + देंगे (denge)
- होम डिलीवरी [hi] home delivery - build from parts: होम (home) + डिलीवरी (delivery)
- आ जाइए [hi] please come - build from parts: आ (aa) + जाइए (jaiye)
- आना पड़ेगा [hi] will have to come - build from parts: आना (aana) + पड़ेगा (padega)
- ले जाइए [hi] take it away - build from parts: ले (le) + जाइए (jaiye)
- दुकान पे [hi] at the shop - build from parts: दुकान (dukan) + पे (pe)
- परसों [hi] day after tomorrow
- नाहीये [mr] is not (colloquial)
- नाय [mr] no (colloquial)
- संपलं [mr] finished (neut.)
- रुपयांना [mr] for rupees
- रुपयाला [mr] per rupee / for a rupee
- पाठवणार [mr] will send
- घेऊन जा [mr] take it away - build from parts: घेऊन (gheun) + जा (ja)
- बोला [mr] speak / go ahead
- बघतो [mr] (I) will check
- मागवतो [mr] (I) will order it
- मागवावा [mr] (it) must be ordered
- नग [mr] piece (count)
- ठीक आहे [mr] okay - build from parts: ठीक (thik) + आहे (ahe)

Multi-word entries carry `parts` (the table entries for each word), so a phrase with no or few
attested spellings can be spelled word by word, e.g. भेज दूंगा = bhej + dunga.

## How the table was built

1. **Dakshina lexicons** (hi, mr; train+dev+test): Devanagari word -> romanization -> number of annotators who wrote it. Exact match on the word list's Devanagari forms.
2. **Dakshina romanized sentences** (hi, mr; 10,000 Wikipedia sentences each, romanized by native speakers; the token-aligned, lower-cased, no-punctuation file): every aligned word or run of words equal to a list form counts its romanization once. Only spellings are kept.
3. **COMI-LINGUA text normalization** (TN_train + TN_test, 29,547 rows, Hindi only): each row has a noisy romanized comment and three annotators' normalized versions. Noisy tokens are aligned to each normalized version; when 2 or 3 of the 3 annotators normalize a noisy word (or 2-word span) to a seed spelling from `word_list.tsv`, that noisy spelling is counted for the word. No sentence text is kept.
4. **Filters** (details in `build_log.json`):
   - COMI spellings seen only once dropped (769 spelling entries), and COMI spellings that are another common word or another list word's seed dropped (2975 occurrences; e.g. `ki`, `ko`, `ke`).
   - Dakshina spellings dropped: 118 (implausible alignment: 62, other sense: 29, another common word: 18, single letter in aligned sentence: 9). 'Other sense' means the same Devanagari string used for an English word in Wikipedia (चेक = Czech / cheque, पीस = peace, कम = come, रोल = role, बस = bus, स्टॉक = stalk). In COMI-LINGUA, `bus` for बस and `prize` for प्राइस are kept: there the annotators read them as बस and price, so writers really used them.
   - Spellings with `q` for a plain क in the Dakshina lexicons (some annotators' habit: qaqa, qampani, paqet) are kept in `variants` with a `note` but left out of `weights` (14 rows).
5. **Independent check**: every (word, spelling) pair was read by a second reviewer; 110 pairs (113 rows) that were another word, another form, an unrelated English word or junk were removed and the weights recomputed with the formula below (`check_report.json`). Re-running `build_variants.py` alone does not apply this step.
6. **Weights**: within each source a spelling's share of that source's count; sources mixed with weight n/sqrt(n+25) (n = the entry's count in that source), so a big source counts more but a source with 1-3 counts cannot dominate. Weights in an entry sum to 1.

## File format

```
{"नहीं": {"lang": "hi", "gloss": "no / not", "category": "refusal",
          "native_forms": ["नहीं", "नही"],
          "variants": [{"roman": "nahi", "count": 4403, "source": "comi-lingua-tn"}, ...],
          "weights": [{"roman": "nahi", "weight": 0.5577}, ...]},
 "भेज दूंगा": {..., "variants": [], "weights": [], "parts": ["भेज", "दूंगा"]}, ...}
```

`variants` keeps one row per (spelling, source) with the raw count; `weights` is the merged, normalized distribution to sample from. All spellings are lower case, letters a-z and spaces only.

## Known problems (read before use)

- **Not phone-call text.** Dakshina is Wikipedia sentences and words romanized by native-speaker annotators; COMI-LINGUA TN is noisy user-written Hinglish comments. Neither is shop talk, so the table says how people spell a word, not how often shopkeepers say it.
- **Marathi is thin.** COMI-LINGUA is Hindi-English only, so Marathi comes from Dakshina alone (Wikipedia vocabulary). Colloquial forms (नाय, नाहीये, बोला, बघतो, मागवतो, पाठवणार, संपलं, नग, रुपयांना) have no attested spelling, and many Marathi words have only 1-3 attestations.
- **Dakshina lexicon oddities.** Isolated-word annotation gives some spellings few people type (pyacket for पॅकेट, hoowe for हो, taaee for ताई). When a word has only lexicon data these get equal weight with the normal spelling (पॅकेट: 8 spellings at 12.5% each). Consider adding the English spelling yourself for loanwords.
- **Alignment noise.** The independent check removed 110 (word, spelling) pairs that were another word or form (kitni under कितना, milti under मिलता, aja under आया, jayegi under जाएगा, bola under बोलो, hogi under होगा), unrelated English words (his, key, take, low, prize) or junk; the list and reasons are in `check_report.json`. Typos of the word itself (hsi for hai, thil for thik) were kept; they carry very small weights.
- **Abbreviations are typing habits.** `h` for है, `rs` for रुपये, `kg` for किलो, `mam` for मैडम are common in typed text; whether a transcript of speech would use them is your choice.
- **Spellings asked about but not found.** `istock` and `dilivery` were not attested; `delevery` and `dilevery` occur once each in the COMI text and fall under the 2-count threshold.
- **हैं vs है.** Writers often spell हैं as `hai` (14%) and है as `h`; this is real usage, not an error.
- **COMI-LINGUA licence statement is inconsistent**: the repository's LICENSE.txt is the CC BY-SA 4.0 text and the README badge says CC BY-SA 4.0, but the README text says CC BY 4.0. This folder is CC BY-SA 4.0, which meets either reading. See NOTICE.md; only spelling counts are taken.

## Reproduce

```
curl -sL https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar \
  | tar -x --wildcards '*/hi/lexicons/*' '*/hi/romanized/*' '*/mr/lexicons/*' '*/mr/romanized/*'
curl -sLO https://raw.githubusercontent.com/lingo-iitgn/COMI-LINGUA/main/TN_train.csv
curl -sLO https://raw.githubusercontent.com/lingo-iitgn/COMI-LINGUA/main/TN_test.csv
python3 -I build_variants.py word_list.tsv dakshina_dataset_v1.0 . OUT_DIR build_log.json
```

Python 3 standard library only; about 35 seconds; output is deterministic.
