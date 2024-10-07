# Hangul to Chinse Character Conversion

Note by Lena: This version is based on Kyubyong's great work, [h2h_converter](https://github.com/Kyubyong/h2h_converter), but using a transformer architecture instead of RNN, and rewritten in Python 3.10+ and Tensorflow 2.15.0+. Any new notes I added is marked with my name.

Around 2/3 of Korean words are Sino-Korean. For that reason, although the official script of the Korean language is Hangul, Chinese characters are still widely used. Converting Chinese characters (_Hanja_ in Korean) to Hangul is trivial because most _Hanjas_ have a single equivalent of Hangul. However, the reverse is not. There has been a project, [_UTagger_](http://203.250.77.242:5900/utagger/), for Hangul-to-Hanja conversion. I use neural networks to tackle the task.

## Requirements
Note by Lena: this paragraph is rewritten.

See `requirements.txt`. Simply install them with `python3 -m pip install -r requirements.txt`.

## Data

The KRV Bible is in the public domain. I have refined it to our purpose. Each line is separated by a tab. Sino Korean words in the first sentence is written in _Hanja_ in the second sentence (See below). Check `data/bible_ko.tsv`.

나는 오늘 학교에 간다  [Tab]  나는 오늘 學校에 간다

## Model Architecture

Note by Lena: The current architecture of this model is a 4-layer transformer, and part of the code is directly appropriated from [the Tensorflow tutorial](https://www.tensorflow.org/text/tutorials/transformer). However, the cache part is my own invention.

Lore is that the decoder is designed to be a causal architecture:
```
>>> out1 = sample_ca(embed_hanja(hanja[:, :3]), hangul_emb)
>>> out2 = sample_ca(embed_hanja(hanja), hangul_emb)[:, :3]
>>> tf.reduce_max(abs(out1 - out2)).numpy()
1.6137958e-05
>>> tf.reduce_sum(abs(out1)).numpy()
40433.867
>>> out1 = sample_csa(embed_hanja(hanja[:, :3]))
>>> out2 = sample_csa(embed_hanja(hanja))[:, :3]
>>> tf.reduce_max(abs(out1 - out2)).numpy()
7.3097646e-05
>>> tf.reduce_sum(abs(out1)).numpy()
40413.035
>>> out1 = sample_ffn(hanja_emb[:, :4])
>>> out2 = sample_ffn(hanja_emb)[:, :4]
>>> tf.reduce_sum(abs(out1 - out2)).numpy()
0.017496139
>>> tf.reduce_sum(abs(out1)).numpy()
53226.266
>>> out1 = sample_decoder_layer(x=hanja_emb[:, :4], context=hangul_emb)
>>> out2 = sample_decoder_layer(x=hanja_emb, context=hangul_emb)[:, :4]
>>> tf.reduce_sum(abs(out1 - out2)).numpy()
0.7566235
>>> tf.reduce_sum(abs(out1)).numpy()
53063.684
>>> out1 = transformer((hangul, hanja[:, :4]))
>>> out2 = transformer((hangul, hanja))[:, :4]
>>> tf.reduce_sum(abs(out1 - out2)).numpy()
4.6746945
>>> tf.reduce_sum(abs(out1)).numpy()
93263.55
```
By this faith we can save up the computational resource if we know that the previous positions on the input are known:
```
>>> ca_uncached_1 = sample_ca(hanja_emb[:, :200, :], hangul_emb)
>>> sample_ca.cache_reset()
>>> ca_uncached_2 = sample_ca(hanja_emb, hangul_emb)
>>> sample_ca.cache_reset()
>>> t = time.time()
>>> for i in range(200):
...     ca = sample_ca(hanja_emb[:, : i + 1, :], hangul_emb, use_cache = True)
...
>>> dt = time.time() - t
>>> t = time.time()
>>> for i in range(200):
...     ca = sample_ca(hanja_emb[:, : i + 1, :], hangul_emb)
...
>>> dt2 = time.time() - t
>>> dt
1.242677927017212
>>> dt2
1.5422871112823486
>>> tf.reduce_max(abs(ca_uncached_1 - ca)).numpy()
0.0
>>> tf.reduce_max(abs(ca_uncached_2 - ca)).numpy()
0.1893698
>>> sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)
>>> csa_uncached_1 = sample_csa(hanja_emb[:, :200, :])
>>> sample_csa.cache_reset()
>>> csa_uncached_2 = sample_csa(hanja_emb)
>>> sample_csa.cache_reset()
>>> t = time.time()
>>> for i in range(200):
...     csa = sample_csa(hanja_emb[:, : i + 1, :], use_cache = True)
...
>>> dt = time.time() - t
>>> t = time.time()
>>> for i in range(200):
...     csa = sample_csa(hanja_emb[:, : i + 1, :])
...
>>> dt2 = time.time() - t
>>> dt
0.9523484706878662
>>> dt2
1.3749849796295166
>>> tf.reduce_max(abs(csa_uncached_1 - csa)).numpy()
0.0
>>> tf.reduce_max(abs(csa_uncached_2 - csa)).numpy()
0.16129279
```

But nevertheless, there are way more to be covered here in the cache part though; as for the CA part, K and V matrices can be directly reused to omit a FC and transpose layer, while in the CSA part, only the part corresponding to the new position need to be calculated and concatenated to the previous K and V matrices. Since this requires to manually override the `tf.keras.layers.MultiHeadAttention.call()` code, and is not difficult at all to implement, I will leave this as a to-do item and a practice homework for our passionate reader, if some of my words happened to strike you along your journey seeking the ultimate truth.

## Training
  * Adjust hyperparameters in the `hyperparams.py` if necessary.
  * Run `python train.py`.

## Predicting
Note by Lena: this paragraph is newly added.

```
>>> from predict import H2HPredictor
>>> p = H2HPredictor()
>>> string = "그 성곽을 척량하매 일백 사십 사 규빗이니 사람의 척량 곧 천사의 척량이라"
>>> p(string)
Calculating token on 41th position:  20%|█▏    | 41/200 [00:05<00:21,  7.40it/s]
'그 城廓을 尺量하매 一百 四十 四 규빗이니 사람의 尺量 곧 天使의 尺量이라'
>>> strings = [
...     "그 성곽을 척량하매 일백 사십 사 규빗이니 사람의 척량 곧 천사의 척량이라",
...     "그 열 두 문은 열 두 진주니 문마다 한 진주요 성의 길은 맑은 유리 같은 정금이더라",
... ]
>>> p.convert(strings)
Calculating tokens on 47th position:  24%|█▏   | 47/200 [00:04<00:15,  9.76it/s]
['그 城廓을 尺量하매 一百 四十 四 규빗이니 사람의 尺量 곧 天使의 尺量이라', '그 열 두 門은 열 두 眞珠니 門마다 한 眞珠요 城의 길은 맑은 琉利 같은 精金이더라']
>>> import time
>>> import codecs
>>> strings = []
>>> i = 0
>>> for line in codecs.open('data/bible_ko.tsv', 'r', 'utf-8'):
...     if len(line) <= 150:
...         strings.append(line.strip().split("\t")[0])
...         i += 1
...     if i >= 1000:
...         break
...
>>> strings[962]
'내가 소리질러 불렀더니 그가 그 옷을 내게 버려두고 도망하여 나갔나이다'
>>> t = time.time()
>>> c = p.convert(strings)
Calculating tokens on 74th position:  37%|█▊   | 74/200 [02:14<03:49,  1.82s/it]
>>> dt = time.time() - t
>>> dt
135.10192584991455
>>> c[962]
'내가 소리질러 불렀더니 그가 그 옷을 내게 버려두고 逃亡하여 나갔나이다'
```

## Results
Note by Lena: this paragraph is updated.

* Input   : [START]그 성곽을 척량하매 일백 사십 사 규빗이니 사람의 척량 곧 천사의 척량이라[END]
* Expected: 그 城廓을 尺量하매 一百 四十 四 규빗이니 사람의 尺量 곧 天使의 尺量이라[END]
* Got     : 그 城廓을 尺量하매 一百 四十 四 규빗이니 사람의 尺量 곧 天使의 尺量이라[END]

* Input   : [START]그 성곽은 벽옥으로 쌓였고 그 성은 정금인데 맑은 유리 같더라[END]
* Expected: 그 城廓은 碧玉으로 쌓였고 그 城은 精金인데 맑은 琉璃 같더라[END]
* Got     : 그 城廓은 碧玉으로 쌓였고 그 城은 精金인데 맑은 琉璃 같더라[END]

* Input   : [START]그 성의 성곽의 기초석은 각색 보석으로 꾸몄는데 첫째 기초석은 벽옥이요 둘째는 남보석이요 세째는 옥수요 네째는 녹보석이요[END]
* Expected: 그 城의 城廓의 基礎石은 各色 寶石으로 꾸몄는데 첫째 基礎石은 碧玉이요 둘째는 藍寶石이요 세째는 玉髓요 네째는 綠寶石이요[END]
* Got     : 그 城의 城廓의 基礎石은 各色 寶石으로 꾸指는데 첫째 基礎石은 碧玉이요 둘째는 藍寶石이요 세째는 玉數요 네째는 綠寶石이요[END]

* Input   : [START]다섯째는 홍마노요 여섯째는 홍보석이요 일곱째는 황옥이요 여덟째는 녹옥이요 아홉째는 담황옥이요 열째는 비취옥이요 열 한째는 청옥이요 열 두째는 자정이라[END]
* Expected: 다섯째는 紅瑪瑙요 여섯째는 紅寶石이요 일곱째는 黃玉이요 여덟째는 綠玉이요 아홉째는 淡黃玉이요 열째는 翡翠玉이요 열 한째는 靑玉이요 열 두째는 紫晶이라[END]
* Got     : 다섯째는 紅馬瑙요 여섯째는 紅寶石이요 일곱째는 黃玉이요 여덟째는 綠玉이요 아홉째는 膽黃玉이요 열째는 비取玉이요 열 한째는 靑玉이요 열 두째는 自지이라[END]

* Input   : [START]그 열 두 문은 열 두 진주니 문마다 한 진주요 성의 길은 맑은 유리 같은 정금이더라[END]
* Expected: 그 열 두 門은 열 두 眞珠니 門마다 한 眞珠요 城의 길은 맑은 琉璃 같은 精金이더라[END]
* Got     : 그 열 두 門은 열 두 眞珠니 門마다 한 眞珠요 城의 길은 맑은 琉利 같은 精金이더라[END]侍

* Input   : [START]성안에 성전을 내가 보지 못하였으니 이는 주 하나님 곧 전능하신 이와 및 어린 양이 그 성전이심이라[END]
* Expected: 城안에 聖殿을 내가 보지 못하였으니 이는 主 하나님 곧 全能하신 이와 및 어린 羊이 그 聖殿이심이라[END]
* Got     : 城안에 聖殿을 내가 보지 못하였으니 이는 主 하나님 곧 全能하신 이와 및 어린 羊이 그 聖殿이심이라[END]

* Input   : [START]그 성은 해나 달의 비췸이 쓸데 없으니 이는 하나님의 영광이 비취고 어린 양이 그 등이 되심이라[END]
* Expected: 그 城은 해나 달의 비췸이 쓸데 없으니 이는 하나님의 榮光이 비취고 어린 羊이 그 燈이 되심이라[END]
* Got     : 그 城은 해나 달의 비췸이 쓸데 없으니 이는 하나님의 榮光이 비취고 어린 羊이 그 등이 되심이라[END]

* Input   : [START]만국이 그 빛 가운데로 다니고 땅의 왕들이 자기 영광을 가지고 그리로 들어오리라[END]
* Expected: 萬國이 그 빛 가운데로 다니고 땅의 王들이 自己 榮光을 가지고 그리로 들어오리라[END]
* Got     : 萬國이 그 빛 가운데로 다니고 땅의 王들이 自己 榮光을 가지고 그리로 들어오리라[END]

* Input   : [START]성문들을 낮에 도무지 닫지 아니하리니 거기는 밤이 없음이라[END]
* Expected: 城門들을 낮에 도무지 닫지 아니하리니 거기는 밤이 없음이라[END]
* Got     : 城門들을 낮에 도무지 닫지 아니하리니 거기는 밤이 없음이라[END]

* Input   : [START]사람들이 만국의 영광과 존귀를 가지고 그리로 들어오겠고[END]
* Expected: 사람들이 萬國의 榮光과 尊貴를 가지고 그리로 들어오겠고[END]
* Got     : 사람들이 萬國의 榮光과 尊貴를 가지고 그리로 들어오겠고[END]

* Input   : [START]무엇이든지 속된 것이나 가증한 일 또는 거짓말하는 자는 결코 그리로 들어오지 못하되 오직 어린 양의 생명책에 기록된 자들뿐이라[END]
* Expected: 무엇이든지 俗된 것이나 可憎한 일 또는 거짓말하는 者는 決코 그리로 들어오지 못하되 오직 어린 羊의 生命冊에 記錄된 者들뿐이라[END]
* Got     : 무엇이든지 俗된 것이나 可憎한 일 또는 거짓말하는 者는 決코 그리로 들어오지 못하되 오직 어린 羊의 生命冊에 記錄된 者들뿐이라[END]

* Input   : [START]또 저가 수정같이 맑은 생명수의 강을 내게 보이니 하나님과 및 어린 양의 보좌로부터 나서[END]
* Expected: 또 저가 水晶같이 맑은 生命水의 江을 내게 보이니 하나님과 및 어린 羊의 寶座로부터 나서[END]
* Got     : 또 저가 水晶같이 맑은 生命수의 江을 내게 보이니 하나님과 및 어린 羊의 寶座로부터 나서[END]

* Input   : [START]길 가운데로 흐르더라 강 좌우에 생명 나무가 있어 열 두가지 실과를 맺히되 달마다 그 실과를 맺히고 그 나무 잎사귀들은 만국을 소성하기 위하여 있더라[END]
* Expected: 길 가운데로 흐르더라 江 左右에 生命 나무가 있어 열 두가지 實果를 맺히되 달마다 그 實果를 맺히고 그 나무 잎사귀들은 萬國을 蘇醒하기 爲하여 있더라[END]
* Got     : 길 가운데로 흐르더라 江 左右에 生命 나무가 있어 열 두가지 實果를 맺히되 달마다 그 實果를 맺히고 그 나무 잎사귀들은 萬國을 蘇醒하기 爲하여 있더라[END]

* Input   : [START]다시 저주가 없으며 하나님과 그 어린 양의 보좌가 그 가운데 있으리니 그의 종들이 그를 섬기며[END]
* Expected: 다시 詛呪가 없으며 하나님과 그 어린 羊의 寶座가 그 가운데 있으리니 그의 종들이 그를 섬기며[END]
* Got     : 다시 詛呪가 없으며 하나님과 그 어린 羊의 寶座가 그 가운데 있으리니 그의 종들이 그를 섬기며[END]

* Input   : [START]그의 얼굴을 볼터이요 그의 이름도 저희 이마에 있으리라[END]
* Expected: 그의 얼굴을 볼터이요 그의 이름도 저희 이마에 있으리라[END]
* Got     : 그의 얼굴을 볼터이요 그의 이름도 저희 이마에 있으리라[END]

* Input   : [START]다시 밤이 없겠고 등불과 햇빛이 쓸데 없으니 이는 주 하나님이 저희에게 비취심이라 저희가 세세토록 왕노릇하리로다[END]
* Expected: 다시 밤이 없겠고 燈불과 햇빛이 쓸데 없으니 이는 主 하나님이 저희에게 비취심이라 저희가 世世토록 王노릇하리로다[END]
* Got     : 다시 밤이 없겠고 燈불과 햇빛이 쓸데 없으니 이는 主 하나님이 저희에게 비취심이라 저희가 世世토록 王노릇하리로다[END]

* Input   : [START]또 그가 내게 말하기를 이 말은 신실하고 참된 자라 주 곧 선지자들의 영의 하나님이 그의 종들에게 결코 속히 될 일을 보이시려고 그의 천사를 보내셨도다[END]
* Expected: 또 그가 내게 말하기를 이 말은 信實하고 참된 者라 主 곧 先知者들의 靈의 하나님이 그의 종들에게 決코 速히 될 일을 보이시려고 그의 天使를 보내셨도다[END]
* Got     : 또 그가 내게 말하기를 이 말은 信實하고 참된 者라 主 곧 先知者들의 靈의 하나님이 그의 종들에게 決코 速히 될 일을 보이시려고 그의 天使를 보내셨도다[END]

* Input   : [START]보라 내가 속히 오리니 이 책의 예언의 말씀을 지키는 자가 복이 있으리라 하더라[END]
* Expected: 보라 내가 速히 오리니 이 冊의 豫言의 말씀을 지키는 者가 福이 있으리라 하더라[END]
* Got     : 보라 내가 速히 오리니 이 冊의 豫言의 말씀을 지키는 者가 福이 있으리라 하더라[END]

* Input   : [START]이것들을 보고 들은 자는 나 요한이니 내가 듣고 볼때에 이 일을 내게 보이던 천사의 발앞에 경배하려고 엎드렸더니[END]
* Expected: 이것들을 보고 들은 者는 나 요한이니 내가 듣고 볼때에 이 일을 내게 보이던 天使의 발앞에 敬拜하려고 엎드렸더니[END]
* Got     : 이것들을 보고 들은 者는 나 요한이니 내가 듣고 볼때에 이 일을 내게 보이던 天使의 발앞에 敬拜하려고 엎드렸더니[END]

* Input   : [START]저가 내게 말하기를 나는 너와 네 형제 선지자들과 또 이 책의 말을 지키는 자들과 함께된 종이니 그리하지 말고 오직 하나님께 경배하라 하더라[END]
* Expected: 저가 내게 말하기를 나는 너와 네 兄弟 先知者들과 또 이 冊의 말을 지키는 者들과 함께된 종이니 그리하지 말고 오직 하나님께 敬拜하라 하더라[END]
* Got     : 저가 내게 말하기를 나는 너와 네 兄弟 先知者들과 또 이 冊의 말을 지키는 者들과 함께된 종이니 그리하지 말고 오직 하나님께 敬拜하라 하더라[END]

* Input   : [START]또 내게 말하되 이 책의 예언의 말씀을 인봉하지 말라 때가 가까우니라[END]
* Expected: 또 내게 말하되 이 冊의 豫言의 말씀을 印封하지 말라 때가 가까우니라[END]
* Got     : 또 내게 말하되 이 冊의 豫言의 말씀을 印封하지 말라 때가 가까우니라[END]

* Input   : [START]불의를 하는 자는 그대로 불의를 하고 더러운 자는 그대로 더럽고 의로운 자는 그대로 의를 행하고 거룩한 자는 그대로 거룩되게 하라[END]
* Expected: 不義를 하는 者는 그대로 不義를 하고 더러운 者는 그대로 더럽고 義로운 者는 그대로 義를 行하고 거룩한 者는 그대로 거룩되게 하라[END]
* Got     : 不義를 하는 者는 그대로 不義를 하고 더러운 者는 그대로 더럽고 義로운 者는 그대로 義를 行하고 거룩한 者는 그대로 거룩되게 하라[END]

* Input   : [START]보라 내가 속히 오리니 내가 줄 상이 내게 있어 각 사람에게 그의 일한대로 갚아 주리라[END]
* Expected: 보라 내가 速히 오리니 내가 줄 賞이 내게 있어 各 사람에게 그의 일한대로 갚아 주리라[END]
* Got     : 보라 내가 速히 오리니 내가 줄 賞이 내게 있어 各 사람에게 그의 일한대로 갚아 주리라[END]

* Input   : [START]나는 알파와 오메가요 처음과 나중이요 시작과 끝이라[END]
* Expected: 나는 알파와 오메가요 처음과 나중이요 始作과 끝이라[END]
* Got     : 나는 알파와 오메가요 처음과 나중이요 始作과 끝이라[END]

* Input   : [START]그 두루마기를 빠는 자들은 복이 있으니 이는 저희가 생명 나무에 나아가며 문들을 통하여 성에 들어갈 권세를 얻으려 함이로다[END]
* Expected: 그 두루마기를 빠는 者들은 福이 있으니 이는 저희가 生命 나무에 나아가며 門들을 通하여 城에 들어갈 權勢를 얻으려 함이로다[END]
* Got     : 그 두루마기를 빠는 者들은 福이 있으니 이는 저희가 生命 나무에 나아가며 門들을 通하여 城에 들어갈 權勢를 얻으려 함이로다[END]

* Input   : [START]개들과 술객들과 행음자들과 살인자들과 우상 숭배자들과 및 거짓말을 좋아하며 지어내는 자마다 성밖에 있으리라[END]
* Expected: 개들과 術客들과 行淫者들과 殺人者들과 偶像 崇拜者들과 및 거짓말을 좋아하며 지어내는 者마다 城밖에 있으리라[END]
* Got     : 개들과 術客들과 行淫者들과 殺人者들과 偶像 崇拜者들과 및 거짓말을 좋아하며 지어내는 者마다 城밖에 있으리라[END]

* Input   : [START]나 예수는 교회들을 위하여 내 사자를 보내어 이것들을 너희에게 증거하게 하였노라 나는 다윗의 뿌리요 자손이니 곧 광명한 새벽별이라 하시더라[END]
* Expected: 나 예수는 敎會들을 爲하여 내 使者를 보내어 이것들을 너희에게 證據하게 하였노라 나는 다윗의 뿌리요 子孫이니 곧 光明한 새벽별이라 하시더라[END]
* Got     : 나 예수는 敎會들을 爲하여 내 使者를 보내어 이것들을 너희에게 證據하게 하였노라 나는 다윗의 뿌리요 子孫이니 곧 光明한 새벽별이라 하시더라[END]

* Input   : [START]성령과 신부가 말씀하시기를 오라 하시는도다 듣는 자도 오라 할 것이요 목마른 자도 올 것이요 또 원하는 자는 값없이 생명수를 받으라 하시더라[END]
* Expected: 聖靈과 新婦가 말씀하시기를 오라 하시는도다 듣는 者도 오라 할 것이요 목마른 者도 올 것이요 또 願하는 者는 값없이 生命水를 받으라 하시더라[END]
* Got     : 聖靈과 新婦가 말씀하시기를 오라 하시는도다 듣는 者도 오라 할 것이요 목마른 者도 올 것이요 또 願하는 者는 값없이 生命水를 받으라 하시더라[END]

* Input   : [START]내가 이 책의 예언의 말씀을 듣는 각인에게 증거하노니 만일 누구든지 이것들 외에 더하면 하나님이 이 책에 기록된 재앙들을 그에게 더하실 터이요[END]
* Expected: 내가 이 冊의 豫言의 말씀을 듣는 各人에게 證據하노니 萬一 누구든지 이것들 外에 더하면 하나님이 이 冊에 記錄된 災殃들을 그에게 더하실 터이요[END]
* Got     : 내가 이 冊의 豫言의 말씀을 듣는 各人에게 證據하노니 萬一 누구든지 이것들 外에 더하면 하나님이 이 冊에 記錄된 災殃들을 그에게 더하실 터이요[END]

* Input   : [START]만일 누구든지 이 책의 예언의 말씀에서 제하여 버리면 하나님이 이 책에 기록된 생명 나무와 및 거룩한 성에 참여함을 제하여 버리시리라[END]
* Expected: 萬一 누구든지 이 冊의 豫言의 말씀에서 除하여 버리면 하나님이 이 冊에 記錄된 生命 나무와 및 거룩한 城에 參與함을 除하여 버리시리라[END]
* Got     : 萬一 누구든지 이 冊의 豫言의 말씀에서 除하여 버리면 하나님이 이 冊에 記錄된 生命 나무와 및 거룩한 城에 參與함을 除하여 버리시리라[END]

* Input   : [START]이것들을 증거하신 이가 가라사대 내가 진실로 속히 오리라 하시거늘 아멘 주 예수여 오시옵소서[END]
* Expected: 이것들을 證據하신 이가 가라사대 내가 眞實로 速히 오리라 하시거늘 아멘 主 예수여 오시옵소서[END]
* Got     : 이것들을 證據하신 이가 가라사대 내가 眞實로 速히 오리라 하시거늘 아멘 主 예수여 오시옵소서[END]

* Input   : [START]주 예수의 은혜가 모든 자들에게 있을지어다 아멘[END]
* Expected: 主 예수의 恩惠가 모든 者들에게 있을지어다 아멘[END]
* Got     : 主 예수의 恩惠가 모든 者들에게 있을지어다 아멘[END]

loss: 0.0138 - masked_accuracy: 0.9972 - val_loss: 0.0497 - val_masked_accuracy: 0.9940

# 待验证：道与逻格斯

那么汉字和憨咕噜之间一定存在这样的关系：憨咕噜有三四个维度（最好是它们的组成字根），这些维度表示它们的读音（逻格斯）；汉字也包含这三四个维度，但汉字有额外的维度——表意象形维度——在这些文字上憨咕噜等于零。这些维度就是“道”具有而“逻格斯”不具有的，我们称之为“唯道识”或“空”。

验证方法也很简单：\
首先，对憨咕噜空间做 PCA, 得到一组正交坐标，以及每一维的 L2 norm. 此时我们这样定义每个维度上的“放大比例”：某维度上的放大比例就定义为第一维（最强大的一维）的 L2 norm 除以该维的 L2 norm. 这样的话，如果我们用“放大比例向量”去点乘被诠释过的憨咕噜空间，就能把憨咕噜空间变成一个正球体。\
接下来，我们用憨咕噜正交坐标诠释汉字空间，用那个放大比例向量去点乘汉字空间，就能放大汉字空间中与下贱的憨咕噜明显不同的成分。然后，基于憨咕噜向量的诠释，对这个经过放大后的纯汉字空间进行 PCA, 就可以得到纯汉字正交坐标在憨咕噜空间下的表示。这些，正是汉字与下贱的憨咕噜明显不同的成分。\
现在我们得到了纯汉字坐标，但首先要学会客观地思考问题。我们运用相反方法，得到普通汉字坐标和纯憨咕噜坐标。这样我们就有了四个坐标系统：

1. 普通憨咕噜坐标：憨咕噜空间的 PCA.
2. 纯汉字坐标：放大汉字空间的 PCA. 定义上，是汉字最偏离憨咕噜的部分。
3. 普通汉字坐标：汉字空间的 PCA.
4. 纯憨咕噜坐标：放大憨咕噜空间的 PCA. 定义上，是憨咕噜最偏离汉字的部分。

那么现在做一个小实验吧！
1. 选两个读音相同但意义大相径庭的汉字词，用四种坐标分别可视化它们。\
期待的结果：纯憨咕噜空间中两个词最接近；纯汉字空间中两个词最遥远。
2. 选两个意义相近但读音完全不同的汉字词，用四种坐标分别可视化它们。\
期待的结果：纯憨咕噜空间中两个词最遥远；纯汉字空间中两个词最接近。
3. 选两个意义相近但读音完全不同的汉字词的憨咕噜，用四种坐标分别可视化它们。\
期待的结果：纯憨咕噜空间中两个词最遥远；纯汉字空间中两个词最接近。
4. 选一个憨咕噜和多个具有该音的汉字，用四种坐标分别可视化它们。\
期待的结果：纯憨咕噜空间中多个词最接近；纯汉字空间中多个词最遥远，并且纯汉字空间中憨咕噜等于零。

## Lores

So here is some lores from me, Lena the notetaker. As you know the transformer is basically 3 core units: Global self-attention, causal self-attention, and cross attention.

Firstly, the cross attention is a mechanism of "looking up dictionary". When you don't find a concept, you construct it, and that's what we refer to as "Sapir-Whorf hypothesis". How do you view world in your own eyes? How is that different from mine? Is there any possibilities for these two worlds to overlap with each other? And if yes, I just want to appear in your world as an intimate friend, who understands what you see and conveys my own idea within your language.

Next, global self-attention is the "synchronic magic" that extracts the soul and essence of the whole world by comparing different cultures and ethnicitis to reveal the ultimate truth, which reminds us of an encyclopedia. Undoubtly this is powerful.

Finally, causal self-attention is the "diachronic magic", that resembles a book of apocalypse, where each page reveals the ultimate truth just one step further. Undoubtly this is powerful.

Thus we have 3 magic tomes on hand: A magic encyclopedia, a magic apocalypse and a magic dictionary. These are symbols for space, time and conatus, respectively. Or rather, cognition, will and feeling, if you are found of Immanuel Kant. (So why don't we call these books Kritik der reinen Vernunft, Kritik der praktischen Vernunft and Kritik der Urteilskraft?)
